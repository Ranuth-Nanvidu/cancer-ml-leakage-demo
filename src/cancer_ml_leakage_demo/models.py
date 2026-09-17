from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_models(random_state: int | None = 42) -> dict:
    """Return {model_name: fresh, untrained estimator}.

    Called once per round in metrics.py so every round trains brand
    new models instead of reusing already-fitted ones.
    """
    return {
        "Naive Bayes": GaussianNB(),
        "Nearest Neighbor": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(criterion="entropy", random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "SVM": SVC(random_state=random_state),
        "Neural Network": MLPClassifier(max_iter=1000, random_state=random_state),
    }