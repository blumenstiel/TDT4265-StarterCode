import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer

# Run comparisons between baseline model (Task 2) and improvements (Task 3)
compare_improved_weight_init = True
compare_improved_sigmoid = True
compare_momentum = True
single_plots = True

if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = False
    use_improved_weight_init = False
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
    train_history, val_history = trainer.train(num_epochs)

    if compare_improved_weight_init:
        # Task 3a
        # Comparing baseline model to improved_weight_init
        print("Comparing baseline model to improved_weight_init")
        use_improved_weight_init = True

        model_improved_weight = SoftmaxModel(
            neurons_per_layer,
            use_improved_sigmoid,
            use_improved_weight_init)
        trainer_improved_weight = SoftmaxTrainer(
            momentum_gamma, use_momentum,
            model_improved_weight, learning_rate, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        )
        train_history_improved_weight, val_history_improved_weight = trainer_improved_weight.train(
            num_epochs)
        use_improved_weight_init = False

        if single_plots:
            plt.figure(figsize=(20, 12))
            plt.subplot(1, 2, 1)
            plt.ylim([0, .5])
            utils.plot_loss(train_history["loss"], "Task 2 Model", npoints_to_average=10)
            utils.plot_loss(train_history_improved_weight["loss"], "Task 3a Model - Improved Weight Initialization",
                            npoints_to_average=10)
            utils.plot_loss(val_history_improved_weight["loss"], "Task 3a Model - Validation Loss",
                            npoints_to_average=10)
            plt.xlabel("Number of Training Steps")
            plt.ylabel("Cross Entropy Loss - Average")
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.ylim([0.85, .99])
            utils.plot_loss(val_history["accuracy"], "Task 2 Model")
            utils.plot_loss(val_history_improved_weight["accuracy"], "Task 3a Model - Improved Weight Initialization")
            plt.xlabel("Number of Training Steps")
            plt.ylabel("Validation Accuracy")
            plt.legend()
            # plt.show()
            plt.savefig("task3a_improved_weight.png")

    if compare_improved_sigmoid:
        # Task 3b
        # Comparing baseline model to improved_sigmoid
        print("Comparing baseline model to improved_sigmoid")
        use_improved_weight_init = True
        use_improved_sigmoid = True

        model_improved_sigmoid = SoftmaxModel(
            neurons_per_layer,
            use_improved_sigmoid,
            use_improved_weight_init)
        trainer_improved_sigmoid = SoftmaxTrainer(
            momentum_gamma, use_momentum,
            model_improved_sigmoid, learning_rate, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        )
        train_history_improved_sigmoid, val_history_improved_sigmoid = trainer_improved_sigmoid.train(
            num_epochs)
        use_improved_weight_init = False
        use_improved_sigmoid = False

        if single_plots:
            plt.figure(figsize=(20, 12))
            plt.subplot(1, 2, 1)
            plt.ylim([0, .5])
            utils.plot_loss(train_history["loss"], "Task 2 Model", npoints_to_average=10)
            utils.plot_loss(train_history_improved_sigmoid["loss"], "Task 3b Model - Improved Weight Init and Sigmoid",
                            npoints_to_average=10)
            utils.plot_loss(val_history_improved_sigmoid["loss"], "Task 3b Model - Validation Loss",
                            npoints_to_average=10)
            plt.xlabel("Number of Training Steps")
            plt.ylabel("Cross Entropy Loss - Average")
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.ylim([0.85, .99])
            utils.plot_loss(val_history["accuracy"], "Task 2 Model")
            utils.plot_loss(val_history_improved_sigmoid["accuracy"],
                            "Task 3b Model - Improved Weight Init and Sigmoid")
            plt.xlabel("Number of Training Steps")
            plt.ylabel("Validation Accuracy")
            plt.legend()
            # plt.show()
            plt.savefig("task3b_improved_sigmoid.png")

    if compare_momentum:
        # Task 3c
        # Comparing baseline model to use_momentum
        print("Comparing baseline model to use_momentum")
        use_momentum = True

        model_momentum = SoftmaxModel(
            neurons_per_layer,
            use_improved_sigmoid,
            use_improved_weight_init)
        trainer_momentum = SoftmaxTrainer(
            momentum_gamma, use_momentum,
            model_momentum, learning_rate, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        )
        train_history_momentum, val_history_momentum = trainer_momentum.train(
            num_epochs)
        use_momentum = False

        if single_plots:
            plt.figure(figsize=(20, 12))
            plt.subplot(1, 2, 1)
            plt.ylim([0, .5])
            utils.plot_loss(train_history["loss"], "Task 2 Model", npoints_to_average=10)
            utils.plot_loss(train_history_momentum["loss"], "Task 3c Model - With Momentum", npoints_to_average=10)
            utils.plot_loss(val_history_momentum["loss"], "Task 3c Model - Valdiation Loss", npoints_to_average=10)
            plt.xlabel("Number of Training Steps")
            plt.ylabel("Cross Entropy Loss - Average")
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.ylim([0.85, .99])
            utils.plot_loss(val_history["accuracy"], "Task 2 Model")
            utils.plot_loss(val_history_momentum["accuracy"], "Task 3c Model - With Momentum")
            plt.xlabel("Number of Training Steps")
            plt.ylabel("Validation Accuracy")
            plt.legend()
            # plt.show()
            plt.savefig("task3c_with_momentum.png")

    if compare_improved_weight_init and compare_momentum:
        # Task 3 all
        # Comparing baseline model to combined improvements
        print("Comparing baseline model to Improved Weight Init and Momentum")
        use_improved_weight_init = True
        use_momentum = True

        model_improved_weight_momentum = SoftmaxModel(
            neurons_per_layer,
            use_improved_sigmoid,
            use_improved_weight_init)
        trainer_improved_weight_momentum = SoftmaxTrainer(
            momentum_gamma, use_momentum,
            model_improved_weight_momentum, learning_rate, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        )
        train_history_improved_weight_momentum, val_history_improved_weight_momentum = trainer_improved_weight_momentum.train(
            num_epochs)
        use_improved_weight_init = False
        use_momentum = False

        if single_plots:
            plt.figure(figsize=(20, 12))
            plt.subplot(1, 2, 1)
            plt.ylim([0, .5])
            utils.plot_loss(train_history["loss"], "Task 2 Model", npoints_to_average=10)
            utils.plot_loss(train_history_improved_weight_momentum["loss"],
                            "Task 3 Model - Improved Weight Init and Momentum", npoints_to_average=10)
            utils.plot_loss(val_history_improved_weight_momentum["loss"], "Task 3 Model - Validation Loss",
                            npoints_to_average=10)
            plt.xlabel("Number of Training Steps")
            plt.ylabel("Cross Entropy Loss - Average")

            plt.subplot(1, 2, 2)
            plt.ylim([0.85, .99])
            utils.plot_loss(val_history["accuracy"], "Task 2 Model")
            utils.plot_loss(val_history_improved_weight_momentum["accuracy"],
                            "Task 3 Model - Improved Weight Init and Momentum")
            plt.xlabel("Number of Training Steps")
            plt.ylabel("Validation Accuracy")
            plt.legend()
            # plt.show()
            plt.savefig("task3_all_improvements.png")

    if compare_improved_weight_init and compare_improved_sigmoid and compare_momentum:
        # Task 3 all
        # Comparing baseline model to combined improvements
        print("Comparing baseline model to all improvements")
        use_improved_weight_init = True
        use_improved_sigmoid = True  # TODO: Not working together with momentum
        use_momentum = True

        model_improved_all = SoftmaxModel(
            neurons_per_layer,
            use_improved_sigmoid,
            use_improved_weight_init)
        trainer_improved_all = SoftmaxTrainer(
            momentum_gamma, use_momentum,
            model_improved_all, learning_rate, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        )
        train_history_improved_all, val_history_improved_all = trainer_improved_all.train(
            num_epochs)

        if single_plots:
            plt.figure(figsize=(20, 12))
            plt.subplot(1, 2, 1)
            plt.ylim([0, .5])
            utils.plot_loss(train_history["loss"], "Task 2 Model", npoints_to_average=10)
            utils.plot_loss(train_history_improved_all["loss"], "Task 3 Model - With all Improvements",
                            npoints_to_average=10)
            utils.plot_loss(val_history_improved_all["loss"], "Task 3 Model - Validation Loss",
                            npoints_to_average=10)
            plt.xlabel("Number of Training Steps")
            plt.ylabel("Cross Entropy Loss - Average")

            plt.subplot(1, 2, 2)
            plt.ylim([0.85, .99])
            utils.plot_loss(val_history["accuracy"], "Task 2 Model")
            utils.plot_loss(val_history_improved_all["accuracy"], "Task 3 Model - With all Improvements")
            plt.xlabel("Number of Training Steps")
            plt.ylabel("Validation Accuracy")
            plt.legend()
            # plt.show()
            plt.savefig("task3_improved_weight_momentum.png")

    plt.figure(figsize=(20, 12))
    plt.subplot(1, 2, 1)
    plt.ylim([0, .5])
    utils.plot_loss(train_history["loss"], "Task 2 Model", npoints_to_average=10)
    if compare_improved_weight_init:
        utils.plot_loss(train_history_improved_weight["loss"], "Task 3a Model - Improved Weight Initialization",
                        npoints_to_average=10)
    if compare_improved_sigmoid:
        utils.plot_loss(train_history_improved_sigmoid["loss"], "Task 3b Model - Improved Weight Init and Sigmoid",
                        npoints_to_average=10)
    if compare_momentum:
        utils.plot_loss(train_history_momentum["loss"], "Task 3c Model - With Momentum", npoints_to_average=10)
    if compare_improved_weight_init and compare_momentum:
        utils.plot_loss(train_history_improved_weight_momentum["loss"],
                        "Task 3 Model - Improved Weight Init and Momentum", npoints_to_average=10)
    if compare_improved_weight_init and compare_improved_sigmoid and compare_momentum:
        utils.plot_loss(train_history_improved_all["loss"], "Task 3 Model - With all Improvements",
                        npoints_to_average=10)
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")

    plt.subplot(1, 2, 2)
    plt.ylim([0.85, .99])
    utils.plot_loss(val_history["accuracy"], "Task 2 Model")
    if compare_improved_weight_init:
        utils.plot_loss(val_history_improved_weight["accuracy"], "Task 3a Model - Improved Weight Initialization")
    if compare_improved_sigmoid:
        utils.plot_loss(val_history_improved_sigmoid["accuracy"], "Task 3b Model - Improved Weight Init and Sigmoid")
    if compare_momentum:
        utils.plot_loss(val_history_momentum["accuracy"], "Task 3c Model - With Momentum")
    if compare_improved_weight_init and compare_momentum:
        utils.plot_loss(val_history_improved_weight_momentum["accuracy"],
                        "Task 3 Model - Improved Weight Init and Momentum")
    if compare_improved_weight_init and compare_improved_sigmoid and compare_momentum:
        utils.plot_loss(val_history_improved_all["accuracy"], "Task 3 Model - With all Improvements")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    # plt.show()
    plt.savefig("task3_combined_comparison.png")
