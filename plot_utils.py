import matplotlib.pyplot as plt


def plot(axis, train, validation, title):
    # We create a list of epoch numbers from 1 to the length of the training set
    epochs = range(1, len(train) + 1)
    # Graph of the training data with a solid blue line
    axis.plot(epochs, train, "b-o", label="Training " + title)
    # Graph of the validation data with a red dashed line
    axis.plot(epochs, validation, "r--o", label="Validation " + title)
    # We set the title of the graph, the X and Y axis labels
    axis.set_title("Training and validation " + title)
    axis.set_xlabel("Epochs")
    axis.set_ylabel(title)
    # We show the legend of the graph
    axis.legend()


def multiplot(history):
    fig, axes = plt.subplots(1, 2)
    fig.set_figwidth(11)
    plot(axes[0], history.history["loss"], history.history["val_loss"], "loss")
    plot(
        axes[1],
        history.history["accuracy"],
        history.history["val_accuracy"],
        "accuracy",
    )
    # We show the graphs on screen
    plt.show()