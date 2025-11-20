import matplotlib.pyplot as plt


def multiplot(history):
    fig, axes = plt.subplots(1, 2)
    fig.set_figwidth(11)
    _plot(axes[0], history.history["loss"], history.history["val_loss"], "loss")
    _plot(
        axes[1],
        history.history["accuracy"],
        history.history["val_accuracy"],
        "accuracy",
    )
    plt.show()


def _plot(axis, train, validation, title):
    epochs = range(1, len(train) + 1)
    axis.plot(epochs, train, "b-o", label="Training " + title)
    axis.plot(epochs, validation, "r--o", label="Validation " + title)
    axis.set_title("Training and validation " + title)
    axis.set_xlabel("Epochs")
    axis.set_ylabel(title)
    axis.legend()
