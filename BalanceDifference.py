# Function the create the 2 differences variables
import pandas as pd

def add_balance_diffs(X):
    X = X.copy()
    X["balanceDiffOrig"] = X["oldbalanceOrg"] - X["newbalanceOrig"]
    X["balanceDiffDest"] = X["newbalanceDest"] - X["oldbalanceDest"]
    return X