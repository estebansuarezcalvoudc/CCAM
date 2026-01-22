import matplotlib.pyplot as plt
import seaborn as sns


def display_confussion_matrix(confussion_matrix) -> None:
    plt.figure(figsize=(6, 5))
    plt.title("Matriz de Confusión")
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
