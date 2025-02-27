"""
    Interface for using neural networks for outcome variables, NOT for making knockoffs. We can save the networks, do hyperparameter search, and then train networks with given set of hyperparameters.
    
    The basic architecture is a few dense layers, either shrinking in layer size or staying the same.
"""
import tensorflow as tf
import pandas as pd
import numpy as np

from typing import Literal, Self

class SimpleNN():
    def __init__(
        self: Self,
        save_root: str,
        save_name: str,
        epochs: int,
        dense_activation: str, # 'relu', 'sigmoid',...
        # Hyper parameters
        first_layer_width: float | int, # int = absolute size, float = proportion of input
        layers: int,
        layer_shrink_factor: float,
        learning_rate: float,
        verbose: int = 0
        ) -> Self:
        """
            Sets parameters
        """
        self.save_root = save_root
        self.save_name = save_name
        self.epochs = epochs
        self.dense_activation = dense_activation
        
        # Hyperparameters
        self.hyperparameters: dict[{
            "first_layer_width": float | int,
            "layers": int,
            "layer_shrink_factor": float,
            "learning_rate": float
        }] = {
            "first_layer_width": first_layer_width,
            "layers": layers,
            "layer_shrink_factor": layer_shrink_factor,
            "learning_rate": learning_rate
        }
        
        self.verbose = verbose
        
        self.network = None
        super().__init__()
    #
    
    def getNetwork(
        self: Self,
        input_length: int,
        hyperparameters: dict[ str, any ]
        ) -> tf.keras.Model:
        """
            Builds a network with architecture from hyperparameters
            
            hyperparameters:
                layers: int
                    Number of dense layers
                first_layer_width: int | float
                    if int, layer width. If float, a multiplier of input_length
                layer_shrink_factor: float
                    Multiplier for layer width
                learning_rate: float
                
        """
        if self.verbose > 1:
            print("# Building SimpleNN network with hyper parameters:")
            for _key, _val in hyperparameters.items():
                print("#   {}: {}".format(_key,_val))
            #
        #
        
        input = tf.keras.layers.Input(
            shape = (input_length,),
            name = 'keras_tensor'
        )
        
        first_layer_width: int | float = hyperparameters[
            'first_layer_width'
        ]
        
        layer_size_float: float
        if isinstance(
            first_layer_width,
            int
        ):
            layer_size_float = float( hyperparameters['first_layer_width'] )
        #
        elif isinstance(
            first_layer_width,
            float
        ):
            layer_size_float = first_layer_width*input_length
        #
        else:
            raise Exception(
                "Unrecognized type for first_layer_width = {}".format(
                    first_layer_width
                )
            )
        #/switch typeof hyperparameters['first_layer_width']
        
        # Keep track of "true" layer size so that we
        #   don't have compounding rounding issues
        
        dense_next = tf.keras.layers.Dense(
            round( layer_size_float ),
            activation = self.dense_activation
        )( input )
        
        # Additional input layers
        for _ in range( 1, hyperparameters['layers'] ):
            layer_size_float = layer_size_float*hyperparameters['layer_shrink_factor']
            dense_next = tf.keras.layers.Dense(
                round( layer_size_float ),
                activation = self.dense_activation
            )( dense_next )
        #
        
        # Output
        output = tf.keras.layers.Dense(
            1,
            activation = 'linear'
        )( dense_next )
        
        model: tf.keras.Model = tf.keras.Model(
            input,
            output
        )
        
        model.compile(
            optimizer = tf.keras.optimizers.Adam(
                learning_rate = hyperparameters[
                    'learning_rate'
                ]
            ),
            loss = tf.keras.losses.MeanSquaredError()
        )
        return model
    #/def getNetwork
    
    def fit(
        self: Self,
        X: np.ndarray,
        y: np.ndarray
        ) -> Self:
        self.network = self.getNetwork(
            input_length = X.shape[1],
            hyperparameters = self.hyperparameters
        )
        
        self.network.fit(
            x = X,
            y = y,
            epochs = self.epochs
        )
        return
    #
    
    def predict( self: Self, X: np.ndarray ) -> np.ndarray:
        return self.network.predict(
            X
        ).reshape( (X.shape[0],) )
    #
    
    def call( self: Self, X: np.ndarray | tf.Tensor ) -> np.ndarray | tf.Tensor:
        return self.network.call( X )
    #
#/class SimpleNN

def get_SimpleNN(
    save_root: str = '',
    save_name: str = '',
    epochs: int = 500,
    dense_activation: str = 'relu', # 'relu', 'sigmoid',...
    # Hyper parameters
    first_layer_width: float | int = 0.25, # int = absolute size, float = proportion of input
    layers: int = 2,
    layer_shrink_factor: float = 0.25,
    learning_rate: float = 0.01,
    verbose: int = 0
    ) -> SimpleNN:
    """
        Applies defaults for SimpleNN, while allowing other values
    """
    return SimpleNN(
        save_root = save_root,
        save_name = save_name,
        epochs = epochs,
        dense_activation = dense_activation,
        first_layer_width = first_layer_width,
        layers = layers,
        layer_shrink_factor = layer_shrink_factor,
        learning_rate = learning_rate,
        verbose = verbose
    )
#/def get_SimpleNN

def fit_SimpleNN(
    X: np.ndarray | pd.DataFrame,
    Xk: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    drop_first: bool = True,
    save_root: str = '',
    save_name: str = '',
    epochs: int = 500,
    dense_activation: str = 'relu', # 'relu', 'sigmoid',...
    # Hyper parameters
    first_layer_width: float | int = 0.25, # int = absolute size, float = proportion of input
    layers: int = 2,
    layer_shrink_factor: float = 0.25,
    learning_rate: float = 0.01,
    verbose: int = 0
    ) -> SimpleNN:
    """
        Initialize and run fit with SimpleNN
    """
    
    model: SimpleNN = get_SimpleNN(
        save_root = save_root,
        save_name = save_name,
        epochs = epochs,
        dense_activation = dense_activation,
        first_layer_width = first_layer_width,
        layers = layers,
        layer_shrink_factor = layer_shrink_factor,
        learning_rate = learning_rate,
        verbose = verbose
    )
    
    if isinstance( X, pd.DataFrame ):
        assert all( X.dtypes.iloc[j] == Xk.dtypes.iloc[j] for j in range( X.shape[1] ) )
        from . import Utilities
        
        if any( dtype == 'category' for dtype in X.dtypes ):
            # TEST: 2025-01-26
            _X = pd.get_dummies(
                X, drop_first = drop_first
            ).to_numpy( dtype = float )
            
            _Xk = pd.get_dummies(
                Xk, drop_first = drop_first
            ).to_numpy( dtype = float )
            
            oheDict_X = Utilities.get_oheDict(
                X = X,
                drop_first = drop_first,
                starting_index = 0,
                starting_ohe_index = 0
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
    
    oheDict = oheDict_X | oheDict_Xk

    X_concat: np.ndarray = np.concatenate(
        [_X, _Xk ], axis = 1
    )
    
    # 2025-01-26: Debug
    if oheDict != {} and False:
        print( oheDict )
        for col, val in oheDict.items():
            print("# {} -> {}".format(col,val))
            if col < X.shape[1]:
                continue
                if isinstance( X, pd.DataFrame ):
                    print( X.iloc[0:5,col])
                else:
                    print( X[0:5,col])
                #
                print( X_concat[0:5, val ] )
            #
            else:
                if isinstance( Xk, pd.DataFrame ):
                    print( Xk.iloc[0:5,col-X.shape[1]])
                else:
                    print( Xk[0:5,col-X.shape[1]])
                #
                print( X_concat[ 0:5, val])
            #
        #
        raise Exception("Check oheDict stuff")
    #
    
    _y: np.ndarray
    if isinstance( y, pd.Series ):
        _y = y.to_numpy()
    #
    else:
        _y = y
    #/if isinstance( y, pd.Series )/else
    
    model.fit( X_concat, _y )
    
    return model
#/def fit_SimpleNN
