import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


if __name__ == "__main__":
    # initialize dicts for results
    train_history = dict()
    val_history = dict()
    list_hidden_neurons = [32, 64, 128]

    # evaluate models for each number of hidden neurons
    for num_hidden_neurons in list_hidden_neurons:
        num_epochs = 50
        learning_rate = .1
        batch_size = 32
        neurons_per_layer = [num_hidden_neurons, 10]
        momentum_gamma = .9
        shuffle_data = True

        # TODO: Change if combination of all improvements is working
        use_improved_sigmoid = True
        use_improved_weight_init = True
        use_momentum = False

        # Load dataset
        X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
        X_train = pre_process_images(X_train)
        X_val = pre_process_images(X_val)
        Y_train = one_hot_encode(Y_train, 10)
        Y_val = one_hot_encode(Y_val, 10)

        model = SoftmaxModel(
            neurons_per_layer,
            use_improved_sigmoid,
            use_improved_weight_init)
        trainer = SoftmaxTrainer(
            momentum_gamma, use_momentum,
            model, learning_rate, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        )
        current_train_history, current_val_history = trainer.train(num_epochs)
        train_history[num_hidden_neurons] = current_train_history
        val_history[num_hidden_neurons] = current_val_history

    plt.figure(figsize=(20, 12))
    plt.subplot(1, 2, 1)
    plt.ylim([0, .5])
    for num_hidden_neurons in list_hidden_neurons:
        utils.plot_loss(train_history[num_hidden_neurons]["loss"], f"Model with {num_hidden_neurons} hidden neurons",
                        npoints_to_average=10)
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")

    plt.subplot(1, 2, 2)
    plt.ylim([0.85, .99])
    for num_hidden_neurons in list_hidden_neurons:
        utils.plot_loss(val_history[num_hidden_neurons]["accuracy"], f"Model with {num_hidden_neurons} hidden neurons")

    plt.xlabel("Number of Training Steps")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    # plt.show()
    plt.savefig("task4ab_num_hidden_neurons.png")
