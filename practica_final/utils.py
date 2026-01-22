import matplotlib.pyplot as plt
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


def get_dataset(number_of_features: int = 8):
    algorithm_globals.random_seed = RANDOM_SEED

    spambase = fetch_ucirepo(id=94)

    X = spambase.data.features
    y = spambase.data.targets

    X = MinMaxScaler().fit_transform(X)

    X_8features = PCA(n_components=number_of_features).fit_transform(X)

    X_subset, _, y_subset, _ = train_test_split(
        X_8features, y, train_size=1000, random_state=RANDOM_SEED, stratify=y
    )

    X_rest, X_test, y_rest, y_test = train_test_split(
        X_subset, y_subset, test_size=0.2, random_state=RANDOM_SEED, stratify=y_subset
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=0.1, random_state=RANDOM_SEED, stratify=y_rest
    )

    return X_train, y_train, X_test, y_test, X_val, y_val


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
