import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the WDBC dataset (the same one the paper uses).
    
    Returns
    -------
    X : DataFrame of the 30 numeric features (mean/se/worst of 10
        measurements per cell nucleus).
    y : Series of labels, where 1 = malignant, 0 = benign.
    """
    raw = load_breast_cancer(as_frame=True)
    X = raw.data
    y = raw.target 
    
    y = y.map({0: 1, 1: 0})
    y.name = "diagnosis"
    return X, y


def simple_split(X, y, test_size, random_state: int | None = None,):
    """A plain, no-tricks train/test split (stratified on the label).

    This is just sklearn's train_test_split, wrapped so the notebooks
    read a little nicer and so every pipeline calls one function.
    """
    return train_test_split(X, y, test_size = test_size, random_state=random_state, stratify=y,)