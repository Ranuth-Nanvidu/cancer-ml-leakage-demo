import pandas as pd
from sklearn.metrics import accuracy_score

from cancer_ml_leakage_demo.models import get_models


def run_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    split_fn,
    test_size: float,
    n_rounds: int = 100,
) -> pd.DataFrame:
    """Run `n_rounds` train/test splits and record each model's accuracy.

    Parameters
    ----------
    split_fn : a function with the signature (X, y, test_size,
        random_state) -> X_train, X_test, y_train, y_test. Pass in
        either `simple_split`, `leaky_split`, or `correct_split`.
    test_size : e.g. 0.2 for an 80-20 split, 0.5 for 50-50, etc.
    n_rounds : how many times to repeat the split + train + test
        cycle. The paper uses 100.

    Returns
    -------
    A long-format DataFrame with one row per (round, model) pair and
    an "accuracy" column - easy to group by model afterwards.
    """
    rows = []
    for round_number in range(n_rounds):
        X_train, X_test, y_train, y_test = split_fn(
            X, y, test_size=test_size, random_state=round_number
        )

        for model_name, model in get_models(random_state=round_number).items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            rows.append(
                {"round": round_number, "model": model_name, "accuracy": accuracy}
            )

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Collapse the long-format results into one row per model, with
    the mean, max ("best case") and min ("worst case") accuracy - the
    same three numbers the paper reports in its figures.
    """
    summary = (
        results.groupby("model")["accuracy"]
        .agg(mean_accuracy="mean", max_accuracy="max", min_accuracy="min")
        .sort_values("mean_accuracy", ascending=False)
        .reset_index()
    )
    return summary
