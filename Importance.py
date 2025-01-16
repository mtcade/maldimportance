import pandas as pd
import numpy as np

from typing import Callable, Literal, Protocol, Self
from abc import abstractmethod

class PredictionModel( Protocol ):
    @abstractmethod
    def fit( self: Self, X: np.ndarray, y: np.ndarray, **kwargs ) -> Self:
        ...
    #
    
    @abstractmethod
    def predict( self: Self, X: np.ndarray ) -> np.ndarray:
        ...
    #
    
    @abstractmethod
    def call( self: Self, X ):
        ...
    #
#/class PredictionModel( Protocol )

def auto_diff(
    model: PredictionModel,
    X: np.ndarray
    ) -> np.ndarray:
    import tensorflow as tf
    
    _X = tf.constant( X )
        
    tape: tf.GradientTape
    with tf.GradientTape() as tape:
        tape.watch( _X )
        y_hat: tf.Tensor = model.call( _X )
    #
    
    return tape.gradient( y_hat, _X ).numpy()
#/def auto_diff

def _get_oheDict(
    X: pd.DataFrame,
    drop_first: bool = True,
    starting_index: int = 0
    ) -> dict[ int, int | list[ int ] ]:
    """
        Result maps original column to single column if numeric, list of OHE columns for categories
        
        If drop_first, it does number of categories minus 1
    """
    counts_adjust: int = 1 if drop_first else 0
    
    ohe_dict: dict[ int, int | list[ int ] ] = {}
    col_iterator: int = starting_index
    for _j in range( X.shape[1] ):
        if X.dtypes.iloc[ _j ] == 'category':
            # OHE columns = value_counts - 1
            value_counts = len( X[ X.columns[_j] ].value_counts() )
            ohe_dict[ _j+starting_index ] = list(
                range(
                    col_iterator,
                    col_iterator + value_counts - counts_adjust
                )
            )
            col_iterator += ( value_counts - counts_adjust )
        #
        else:
            # Numeric, one column
            ohe_dict[ _j+starting_index ] = col_iterator
            col_iterator += 1
        #/if X.dtypes[ _j ] == 'category'/else
    #/for _j in range( X.shape[1] )
    return ohe_dict
#/def _get_oheDict

def _localGrad_forCategories(
    j: list[ int ],
    X: np.ndarray,
    model: PredictionModel,
    drop_first: bool
    ) -> np.ndarray:
    """
        Get the change in prediction for a group of category columns by changing each of the values to. 1, and the others to 0
        
        If drop_first, it gets compared with setting all to 0.
    """
    _X: np.ndarray = np.copy( X )
    
    y_out_dim: int = len( j )
    if drop_first: y_out_dim += 1
    
    y_out: np.ndarray = np.zeros( shape = (X.shape[0], y_out_dim ) )
    
    # Get the predicted values at each test category
    for h in range( len(j) ):
        # Reset all to 0, set one to 1
        _X[ :, j ] = 0
        _X[ :, j[h] ] = 1
        
        y_out[ :, h ] = model.predict( _X )
    #
    
    # Use last index setting all to 0
    if drop_first:
        _X[ :, j ] = 0
        y_out[ :, -1 ] = model.predict( _X )
    #
    
    return np.max( y_out, axis = 1 ) - np.min( y_out, axis = 1 )
#/def _localGrad_forCategories

def importancesFromModel(
    model: PredictionModel,
    X: np.ndarray | pd.DataFrame,
    Xk: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    exponent: float = 2.0,
    drop_first: bool = True,
    **kwargs
    ) -> np.ndarray:
    """
        Takes an initialized PredictionModel, fits it, and gets the local gradient importance
        
        X: data
        Xk: knockoff data
        y: outcome data
        bandwidth: amount to perturb X[j] and Xk[j], defaulting to (n**(-0.2)), where
            n is sample size, or X.shape[0]
        exponent: Power to take of the absolute value of local gradients
        drop_first: Passes to `pd.get_dummies` when categorical variables are present. Likely should be true unless `model` can handle the perfect colinearity (neural networks for example).
        **kwargs: passed to `model.fit`
        
        returns W stats, from the absolute difference
    """
    assert X.shape == Xk.shape
    
    # Check for categorical variables, one hot encode variables if necessary
    
    oheDict_X: dict[ int, int | list[ int ] ]
    oheDict_Xk: dict[ int, int | list[ int ] ]
    _X: np.ndarray
    _Xk: np.ndarray
    
    if isinstance( X, pd.DataFrame ):
        assert all( X.dtypes.iloc[j] == Xk.dtypes.iloc[j] for j in range( X.shape[1] ) )
        if any( dtype == 'category' for dtype in X.dtypes ):
            oheDict_X = _get_oheDict(
                X = X, drop_first = drop_first, starting_index = 0
            )
            oheDict_Xk = _get_oheDict(
                X = Xk, drop_first = drop_first, starting_index = X.shape[1]
            )
            
            _X = pd.get_dummies(
                X, drop_first = drop_first
            ).to_numpy( dtype = float )
            
            _Xk = pd.get_dummies(
                Xk, drop_first = drop_first
            ).to_numpy( dtype = float )
        #
        else:
            oheDict_X = {}
            oheDict_Xk = {}
            _X = X.to_numpy()
            _Xk = Xk.to_numpy()
        #/if any( dtype == 'category' for dtype in X.dtypes )
    #/if isinstance( X, pd.DataFrame )
    else:
        oheDict_X = {}
        oheDict_Xk = {}
        _X = X
        _Xk = Xk
    #/if isinstance( X, pd.DataFrame )/else
    
    oheDict: dict[ int, int | list[int] ] = oheDict_X | oheDict_Xk
    
    X_concat: np.ndarray = np.concatenate(
        [_X, _Xk ], axis = 1
    )
    
    _y: np.ndarray
    if isinstance( y, pd.Series ):
        _y = y.to_numpy()
    #
    else:
        _y = y
    #/if isinstance( y, pd.Series )/else
    
    model.fit( X_concat, _y, **kwargs )
    auto_diff_matrix: np.ndarray = auto_diff(
        model,
        X_concat
    )

    p_out: int = X.shape[0], X.shape[1] + Xk.shape[1]
    localGrad_matrix: np.ndarray

    if oheDict == {}:
        # all numeric
        localGrad_matrix = auto_diff_matrix
    #/if oheDict == {}
    else:
        # Some categories
        localGrad_matrix: np.ndarray = np.zeros(
            shape = ( X.shape[0], p_out )
        )
        for j in range( p_out ):
            if isinstance( oheDict[j], int ):
                # numeric
                localGrad_matrix[:,j] = auto_diff_matrix[:, oheDict[j] ]
            #
            else:
                # category
                localGrad_matrix[:,j] = _localGrad_forCategories(
                    j = oheDict[j],
                    X = X_concat,
                    model = model,
                    drop_first = drop_first
                )
            #/if isinstance( oheDict[j], int )/else
        #/for j in range( p_out )
    #/if oheDict == {}/else
    
    importances: np.ndarray = np.mean(
        np.abs( localGrad_matrix )**exponent,
        axis = 0
    )
    return importances
#/def importancesFromModel

def wFromImportances(
    importances: np.ndarray,
    W_method: Literal['difference','signed_max'] = 'difference'
    ) -> np.ndarray:
    p: int = len( importances ) // 2
    W_out: np.ndarray
    if W_method == 'difference':
        W_out = importances[ : p ] - importances[ p: ]
    #
    elif W_method == 'signed_max':
        W_out = np.zeros( shape = ( p,) )
        for j in range(p):
            if importances[ j ] > importances[ j+p ]:
                W_out[ j ] = importances[ j ]
            #
            elif importances[ j ] < importances[ j+p ]:
                W_out[ j ] = importances[ j+p ]
            #/switch importances[ j ] - importances[ j+p ]
        #/for j in range(p)
    else:
        raise Exception("Unrecognized W_method={}".format(W_method))
    #
    return W_out
#/def wFromImportances

def wFromModel(
    model: PredictionModel,
    X: np.ndarray | pd.DataFrame,
    Xk: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    W_method: Literal['difference','signed_max'] = 'difference',
    exponent: float = 2.0,
    drop_first: bool = True,
    **kwargs
    ) -> np.ndarray:
    
    importances: np.ndarray = importancesFromModel(
        X = X,
        Xk = Xk,
        y = y,
        exponent = exponent,
        drop_first = drop_first,
        **kwargs
    )
    
    return wFromImportances(
        importances = importances,
        W_method = W_method
    )
#/def wFromModel
