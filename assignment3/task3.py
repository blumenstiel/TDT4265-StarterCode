import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
import torch
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy
from task2 import ExampleModel

class StandardModel(ExampleModel):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)

            First Model for Task 3a with increased number of neurons
        """
        super().__init__(image_channels, num_classes)
        num_filters = 64  # Set number of filters in first conv layer
        self.num_classes = num_classes
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(64),
            nn.Conv2d(
              in_channels = num_filters,
              out_channels = 128,
              kernel_size = 5,
              stride = 1,
              padding = 2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(128),
            nn.Conv2d(
              in_channels = 128,
              out_channels = 128,
              kernel_size = 5,
              stride = 1,
              padding = 2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(128),
        )
        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        self.num_output_features = 4*4*128
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 10)
        )


class DropoutModel(ExampleModel):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)

        Second Model for Task 3a with increased number of neurons and dropout layer
        """
        super().__init__(image_channels, num_classes)
        num_filters = 64  # Set number of filters in first conv layer
        self.num_classes = num_classes
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(64),
            nn.Dropout(0.2),
            nn.Conv2d(
              in_channels = num_filters,
              out_channels = 128,
              kernel_size = 5,
              stride = 1,
              padding = 2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(128),
            nn.Dropout(0.2),
            nn.Conv2d(
              in_channels = 128,
              out_channels = 128,
              kernel_size = 5,
              stride = 1,
              padding = 2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(128),
            nn.Dropout(0.2),
        )
        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        self.num_output_features = 4*4*128
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 10)
        )


class MomentumTrainer(Trainer):
    """
    Trainer for Task3e with momentum
    """
    def __init__(self, *args, **kwargs):
        super(FinalTrainer, self).__init__(*args, **kwargs)
        # Define our optimizer. SGD = Stochastich Gradient Descent
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                        self.learning_rate,
                                         momentum=0.7)


def create_plots(trainer_list, trainer_labels, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    for trainer, label in zip(trainer_list, trainer_labels):
        utils.plot_loss(trainer.train_history["loss"], label=f"Training loss {label}", npoints_to_average=10)
        utils.plot_loss(trainer.validation_history["loss"], label=f"Validation loss {label}")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    for trainer, label in zip(trainer_list, trainer_labels):
        utils.plot_loss(trainer.validation_history["accuracy"], label=f"Validation Accuracy {label}")
    plt.legend()
    plt.savefig(f"{name}.png")
    plt.show()


def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    epochs = 10
    batch_size = 64
    learning_rate = 0.1
    early_stop_count = 4

    # initialize dataloaders with and without data augmentation
    dataloaders = load_cifar10(batch_size)
    dataloaders_data_augmentation = load_cifar10(batch_size, data_augmentation=0.3)

    base_model = StandardModel(image_channels=3, num_classes=10)
    base_trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        base_model,
        dataloaders
    )
    print("Start training base model Task 2")
    base_trainer.train()

    # Task 3a training a model with increased number of hidden neurons, batch normalization and data augmentation
    standard_model = StandardModel(image_channels=3, num_classes=10)
    trainer_with_data_augmentation = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        standard_model,
        dataloaders_data_augmentation
    )
    print("Start training model with data augmentation")
    trainer_with_data_augmentation.train()

    # Task 3a training a model with increased number of hidden neurons, batch normalization and additional dropout
    # layers
    dropout_model = DropoutModel(image_channels=3, num_classes=10)
    trainer_with_dropout = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        dropout_model,
        dataloaders
    )
    print("Start training model with dropout")
    trainer_with_dropout.train()

    create_plots([trainer_with_data_augmentation], [""], "task3b_best_model")
    create_plots([base_trainer, trainer_with_dropout], ["Task 2", "with Data Augmentation"], "task3d_compare_best_model")

    # reset dropout_model
    dropout_model = DropoutModel(image_channels=3, num_classes=10)
    trainer_with_momentum = MomentumTrainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        dropout_model,
        dataloaders_data_augmentation
    )
    print("Start training model with momentum")
    trainer_with_momentum.train()

    create_plots([trainer_with_momentum], [""], "task3e_final_model")


if __name__ == "__main__":
    main()
