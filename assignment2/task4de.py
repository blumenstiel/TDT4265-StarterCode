import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


if __name__ == "__main__":
    # initialize dicts for results
    train_history = dict()
    val_history = dict()
    list_neurons_per_layer = [[64, 10],
                              [60, 60, 10],
                              [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 10],
                              ]
    model_names = ['Baseline - One hidden layer with 64 neurons',
                   'Model Task 4d - Two hidden layers with 60 neurons',
                   'Model Task 4e - Ten hidden layers with 64 neurons']

    # evaluate models for each number of hidden neurons
    for neurons_per_layer, model_name in zip(list_neurons_per_layer, model_names):
        num_epochs = 50
        learning_rate = .1
        batch_size = 32
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
        train_history[model_name] = current_train_history
        val_history[model_name] = current_val_history

    plt.figure(figsize=(16, 10))
    plt.subplot(1, 2, 1)
    plt.ylim([0, .5])
    for model_name in train_history.keys():
        utils.plot_loss(train_history[model_name]["loss"], model_name,
                        npoints_to_average=10)
    for model_name in train_history.keys():
        utils.plot_loss(val_history[model_name]["loss"], f'Validation {model_name[10:]}',
                        npoints_to_average=10)
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.ylim([0.9, .99])
    for model_name in val_history.keys():
        utils.plot_loss(val_history[model_name]["accuracy"], model_name)

    plt.xlabel("Number of Training Steps")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    # plt.show()
    plt.savefig("task4de_hidden_layers.png")
