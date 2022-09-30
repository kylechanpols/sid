import pandas as pd
import dask.dataframe as dd
import os
import numpy as np

main_path = os.getcwd()
combined = dd.read_csv(os.path.join(main_path,"preds", "trans_efnet_rgb_preds_*.csv"))
combined = combined.compute()
combined = combined.drop(columns=['Unnamed: 0'])

def gini(x: pd.Series) -> float:
    '''
    Description:
    A **bruteforce** function to compute the gini coefficient.
    It computes the MAD of a distribution, and then reweights it with the relative MAD.
    Input: 
    x (pd.Series) 
        - a pandas series representing a predicted distribution.
    
    Output: 
    g (float)
        A float point value between -1 and 1 representing the degree of income inequality.
        -1 means absolutely no inequality and 1 means full inequality.
    Time Complexity: O(n**2)
    Space Complexity: O(n**2)
    Do NOT use for very large samples, e.g. n>10000.
    '''
    x = x.to_numpy(dtype=np.float32)

    x = x[np.logical_not(np.isnan(x))]
    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g


# compute image statistics

combined = combined.groupby(['City','Year'], as_index=False).agg({'pred':['sum',"min","max",'mean','median','std',gini, 'count']})
combined.columns = combined.columns.map('_'.join).str.strip('_') #rename columns to be pred_*, where * is the statistic computed
combined.to_csv(os.path.join(main_path,"preds", "trans_efnet_rgb_preds_image_statistics.csv"), index=False)