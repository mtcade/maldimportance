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

def _localGrad_forNumeric(
    j: int,
    X: np.ndarray,
    y_hat: np.ndarray,
    model: PredictionModel,
    bandwidth: float
    ) -> np.ndarray:
    """
        Get the bandwidth local gradient approximation for variable j
        X: all data
        y_hat: The base prediction, result of `model.predict(X)`
        model: PredictionModel already fit and trained
        bandwidth: Exact literal value
    """
    # Set the X + bandwidth matrix
    X_epsilon: np.ndarray = np.copy( X )
    X_epsilon[:, j ] += bandwidth
    
    # Get the approximation of local gradient via the definition of the derivative
    return ( model.predict(X_epsilon) - y_hat )/bandwidth
#/def _localGrad_forNumeric

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
    local_grad_method: Literal['auto_diff','bandwidth'] = 'auto_diff',
    fit: bool = True,
    bandwidth: float | None = None,
    exponent: float = 2.0,
    drop_first: bool = True,
    verbose: int = 0
    ) -> np.ndarray:
    """
        Takes an initialized PredictionModel, fits it, and gets the local gradient importance
        
        X: data
        Xk: knockoff data
        y: outcome data
        local_grad_method: Whether to use autodifferentiation or a bandwidth method
        bandwidth: amount to perturb X[j] and Xk[j], defaulting to (n**(-0.2)), where
            n is sample size, or X.shape[0]
            Used only when `local_grad_method == 'bandwidth'`
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
        from . import Utilities
        
        if any( dtype == 'category' for dtype in X.dtypes ):
            _X = pd.get_dummies(
                X,
                drop_first = drop_first
            ).to_numpy( dtype = float )
            
            _Xk = pd.get_dummies(
                Xk,
                drop_first = drop_first
            ).to_numpy( dtype = float )
            
            oheDict_X = Utilities.get_oheDict(
                X = X,
                drop_first = drop_first,
                starting_index = 0
            )
            oheDict_Xk = Utilities.get_oheDict(
                X = Xk,
                drop_first = drop_first,
                starting_index = X.shape[1],
                starting_ohe_index = _X.shape[1]
            )
            
            
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
    
    # Fit if necessary; we can have a model already trained
    #   by setting to False
    if fit:
        model.fit( X_concat, _y, )
    #/if fit
    
    auto_diff_matrix: np.ndarray | None
    y_hat: np.ndarray | None
    if local_grad_method == 'auto_diff':
        auto_diff_matrix: np.ndarray = auto_diff(
            model,
            X_concat
        )
        y_hat = None
    #
    elif local_grad_method == 'bandwidth':
        auto_diff_matrix = None
        y_hat = model.predict( X_concat )
        
        if bandwidth is None:
            bandwidth = X_concat.shape[0]**(-0.2)
        #/if bandwidth is None
    #
    else:
        raise Exception("Unrecognized local_grad_method={}".format(local_grad_method))
    #/switch local_grad_method

    p_out: int = X.shape[1] + Xk.shape[1]
    localGrad_matrix: np.ndarray

    if oheDict == {}:
        # all numeric
        if local_grad_method == 'auto_diff':
            localGrad_matrix = auto_diff_matrix
        #
        elif local_grad_method == 'bandwidth':
            localGrad_matrix = np.zeros(
                shape = X_concat.shape
            )
            for j in range( X_concat.shape[1] ):
                localGrad_matrix[ :, j ] = _localGrad_forNumeric(
                    j = j,
                    X = X_concat,
                    y_hat = y_hat,
                    model = model,
                    bandwidth = bandwidth
                )
            #
        #
        else:
            raise Exception("Unrecognized local_grad_method={}".format(local_grad_method))
        #/switch local_grad_method
    #/if oheDict == {}
    else:
        # Some categories
        localGrad_matrix: np.ndarray = np.zeros(
            shape = ( X.shape[0], p_out )
        )
        for j in range( p_out ):
            if isinstance( oheDict[j], int ):
                # numeric
                if local_grad_method == 'auto_diff':
                    localGrad_matrix[:,j] = auto_diff_matrix[:, oheDict[j] ]
                #
                elif local_grad_method == 'bandwidth':
                    localGrad_matrix[ :, j ] = _localGrad_forNumeric(
                        j = oheDict[j],
                        X = X_concat,
                        y_hat = y_hat,
                        model = model,
                        bandwidth = bandwidth
                    )
                #
                else:
                    raise Exception("Unrecognized local_grad_method={}".format(local_grad_method))
                #/switch local_grad_method
            #
            else:
                # category
                # TEST: 2025-01-26
                if False:
                    print("# {} -> {}".format(j,oheDict[j]))
                    
                    if j < X.shape[1]:
                        print(" -- X:")
                        print( X.iloc[0:5, j])
                    #
                    else:
                        print(" -- Xk:")
                        print( Xk.iloc[0:5,j-X.shape[1]] )
                    #
                    print(" -- OHE:")
                    print( X_concat[0:5, oheDict[j]] )
                #
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
    # TEST: 2025-01-26
    if False:
        print( importances )
        raise Exception("Check importances")
    #
    return importances
#/def importancesFromModel

def wFromImportances(
    importances: np.ndarray,
    W_method: Literal['difference','signed_max'] = 'difference',
    verbose: int = 0
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
    local_grad_method: Literal['auto_diff','bandwidth'] = 'auto_diff',
    fit: bool = True,
    exponent: float = 2.0,
    bandwidth: float | None = None,
    drop_first: bool = True,
    verbose: int = 0
    ) -> np.ndarray:
    
    importances: np.ndarray = importancesFromModel(
        model = model,
        X = X,
        Xk = Xk,
        y = y,
        local_grad_method = local_grad_method,
        fit = fit,
        bandwidth = bandwidth,
        exponent = exponent,
        drop_first = drop_first,
        verbose = verbose
    )
    
    return wFromImportances(
        importances = importances,
        W_method = W_method
    )
#/def wFromModel
