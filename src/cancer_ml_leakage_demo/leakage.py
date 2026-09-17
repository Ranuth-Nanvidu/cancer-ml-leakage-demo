import pandas as pd
from cancer_ml_leakage_demo.data import simple_split


def duplicate_dataset(X, y, copies: int = 2) -> tuple[pd.DataFrame, pd.Series]:
    """Stack `copies` exact copies of the dataset on top of each other.

    With copies=2 this is the "doubling the dataset" from the paper.
    """
    X_dup = pd.concat([X] * copies, ignore_index=True)
    y_dup = pd.concat([y] * copies, ignore_index=True)
    return X_dup, y_dup


def leaky_split(X, y, test_size, random_state: int | None = None, copies: int = 2,):
    """The paper's approach: duplicate FIRST, split SECOND.

    Bug: duplicate rows can be split across train and test, so the
    model is tested on data it has effectively already seen.
    """
    X_dup, y_dup = duplicate_dataset(X, y, copies=copies)
    return simple_split(X_dup, y_dup, test_size=test_size, random_state=random_state)


def correct_split(X, y, test_size, random_state: int | None = None, copies: int = 2,):
    """The fix: split FIRST, duplicate SECOND (training half only).

    The test set is untouched original data the model has never seen
    a copy of, so there is no leakage. `copies` is kept as a parameter
    only so this can be compared fairly against leaky_split() - it
    duplicates the training set to roughly match its size, but that
    duplication can never reach the test set.
    """
    X_train, X_test, y_train, y_test = simple_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, y_train = duplicate_dataset(X_train, y_train, copies=copies)
    return X_train, X_test, y_train, y_test