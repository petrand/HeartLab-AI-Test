import numpy as np
import pandas as pd

def distribution(dataset, train_set, val_set, test_set, train_prop=0.7, val_prop=0.15, test_prop=0.15):
    n = len(dataset)
    assert ((len(train_set)/n + 0.05 > train_prop) and (len(train_set)/n - 0.05 < train_prop)), "FAILED TEST: train set is not the expected proportion"
    assert ((len(val_set)/n + 0.05 > val_prop) and (len(val_set)/n - 0.05 < val_prop)), "FAILED TEST: train set is not the expected proportion"
    assert ((len(test_set)/n + 0.05 > test_prop) and (len(test_set)/n - 0.05 < test_prop)), "FAILED TEST: train set is not the expected proportion"
    return "PASSED TEST: Distribution of train/val/test sets"



def normalisation(X):
    assert (((X > 1).sum()  == 0) and ((X < 0).sum()  == 0)), "FAILED TEST: X values are not normalised, values are out of [0,1] bound."
    return "PASSED TEST: Normalisation"



def dimension(set_pair):
    X, y = set_pair[0], set_pair[1]
    assert (len(X.shape)==4), "FAILED TEST: Missing color vector"
    assert (X.shape[0] > 0) and (X.shape[1] == 256) and (X.shape[2] == 256) and (X.shape[3] == 1), "FAILED TEST: Incorrect vector dimensions or empty vector"
    return "PASSED TEST: Dimension"