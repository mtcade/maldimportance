import pandas as pd

def get_oheDict(
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
#/def get_oheDict
