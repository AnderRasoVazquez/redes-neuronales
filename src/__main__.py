"""This module handles the command line interface of the application."""

from argparse import ArgumentParser
from neural_networks import AutopsiesNeuralNetwork
import matplotlib.pyplot as plt


def get_parser():
    """Create an argument parser."""
    parser = ArgumentParser()
    parser.add_argument('--data_path',
                        type=str,
                        help='Path of the data to read.',
                        required=True)

    parser.add_argument('--num_logits',
                        type=int,
                        help='Number of inputs of the neural network. >=1',
                        default=30,
                        required=False)

    parser.add_argument('--num_intermediate',
                        type=int,
                        help='Number of inputs of the neural network intermediate layers. >=1',
                        default=256,
                        required=False)

    parser.add_argument('--num_layers',
                        type=int,
                        help='Number of hidden layers. >=1',
                        default=1,
                        required=False)

    parser.add_argument('--kfold',
                        type=int,
                        help='Number of folds for the kfold evaluation. >=2',
                        default=10,
                        required=False)

    parser.add_argument('--epochs',
                        type=int,
                        help='Number of iterations of the training for the neural network. >=1',
                        default=20,
                        required=False)

    parser.add_argument('--activation_intermediate',
                        type=str,
                        help='Activation function for hidden layers.',
                        default="sigmoid",
                        choices=["sigmoid", "relu", "softmax"],
                        required=False)

    parser.add_argument('--activation_output',
                        type=str,
                        help='Activation function for hidden layers.',
                        choices=["sigmoid", "relu", "softmax"],
                        default='softmax',
                        required=False)

    parser.add_argument('--optimizer',
                        type=str,
                        help='Neural network optimizer algorithm.',
                        choices=["adam", "sgd"],
                        default='adam',
                        required=False)

    parser.add_argument('--loss',
                        type=str,
                        help='Loss function.',
                        choices=["categorical_crossentropy", "binary_crossentropy"],
                        default="binary_crossentropy",
                        required=False)

    parser.add_argument('--verbose',
                        type=int,
                        help='Show more verbose output.',
                        choices=[0, 1, 2],
                        default=1,
                        required=False)

    parser.add_argument('--output_file',
                        type=str,
                        help='File where save results.',
                        default="output.txt",
                        required=False)
    parser.add_argument('--class_attribute',
                        type=str,
                        help='Class to predict.',
                        default="gs_text34",
                        choices=["gs_text34", "module", "site"],
                        required=False)
    parser.add_argument('--plot_path',
                        type=str,
                        help='Path to save plot.',
                        required=False)
    return parser


def create_plot(name, history, key, title, path=None):
    plt.figure(figsize=(16, 10))

    val = plt.plot(history.epoch, history.history['val_' + key],
                   '--', label=name.title() + ' Val')
    plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title() + ' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()
    plt.title(title)

    plt.xlim([0, max(history.epoch)])
    if path:
        plt.savefig(path)
    else:
        plt.show()


def main():
    """Starting point of the application."""
    args = get_parser().parse_args()
    history = AutopsiesNeuralNetwork(**args.__dict__).run()
    create_plot("Classifier", history, args.loss, title=args.class_attribute, path=args.plot_path)


if __name__ == "__main__":
    main()
