import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from qiskit_machine_learning.utils import algorithm_globals
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from ucimlrepo import fetch_ucirepo

RANDOM_SEED = 11


def display_confussion_matrix(
    confussion_matrix, title: str = "Matriz de Confusión"
) -> None:
    plt.figure(figsize=(6, 5))
    plt.title(title)
    sns.heatmap(
        confussion_matrix,
        annot=True,
        fmt="d",  # "d" es para enteros, evita notación científica
        cmap="Reds",  # Mapa de color rojo
        xticklabels=["Legítimo", "Spam"],
        yticklabels=["Legítimo", "Spam"],
    )
    plt.ylabel("Real")
    plt.xlabel("Predicción")
    display(plt.show())


def show_metrics(confussion_matrix) -> None:
    tn, fp, fn, tp = confussion_matrix.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")


def get_dataset(
    number_of_features: int = 8, dataset_size: int = 1000, test_size: float = 0.2
):
    algorithm_globals.random_seed = RANDOM_SEED

    spambase = fetch_ucirepo(id=94)

    X = spambase.data.features
    y = spambase.data.targets

    X = MinMaxScaler().fit_transform(X)

    X_8features = PCA(n_components=number_of_features).fit_transform(X)

    X_subset, _, y_subset, _ = train_test_split(
        X_8features, y, train_size=dataset_size, random_state=RANDOM_SEED, stratify=y
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_subset,
        y_subset,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=y_subset,
    )

    return X_train, y_train, X_test, y_test


def evaluate_classifier(classifier, X_train, y_train, X_test, y_test) -> None:
    print("--- Resultados para entrenamiento ---")
    y_pred = classifier.predict(X_train)
    cm = confusion_matrix(y_train, y_pred)
    show_metrics(cm)

    print("--- Resultados para test ---")
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    show_metrics(cm)
    display_confussion_matrix(cm, title="Matriz de Confusión (test)")


class ExperimentLogger:
    def __init__(self):
        self._results = []

    def log(
        self,
        feature_map_name,
        ansatz_name,
        optimizer_name,
        classifier,
        X_train,
        y_train,
        X_test,
        y_test,
    ):
        y_pred_train = classifier.predict(X_train)
        acc_train = accuracy_score(y_train, y_pred_train)
        prec_train = precision_score(y_train, y_pred_train, zero_division=0)

        y_pred_test = classifier.predict(X_test)
        acc_test = accuracy_score(y_test, y_pred_test)
        prec_test = precision_score(y_test, y_pred_test, zero_division=0)

        self._results.append(
            {
                "Feature Map": feature_map_name,
                "Ansatz": ansatz_name,
                "Optimizer": optimizer_name,
                "Accuracy (train)": acc_train,
                "Precision (train)": prec_train,
                "Accuracy (test)": acc_test,
                "Precision (test)": prec_test,
            }
        )

        print(
            f"--- Resultados para {feature_map_name} + {ansatz_name} + {optimizer_name} ---"
        )
        print(f"Train - Accuracy: {acc_train:.4f}, Precision: {prec_train:.4f}")
        print(f"Test  - Accuracy: {acc_test:.4f}, Precision: {prec_test:.4f}")

        cm = confusion_matrix(y_test, y_pred_test)
        display_confussion_matrix(cm, title=f"Matriz de confusión (Test)")

    @property
    def results_df(self):
        return pd.DataFrame(self._results)
