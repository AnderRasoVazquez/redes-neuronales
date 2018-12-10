"""This module handles the command line interface of the application."""

from argparse import ArgumentParser
from neural_networks import AutopsiesNeuralNetwork


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
                        choices=["sigmoid"],
                        required=False)

    parser.add_argument('--activation_output',
                        type=str,
                        help='Activation function for hidden layers.',
                        choices=["softmax"],
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
                        choices=["categorical_crossentropy"],
                        default="categorical_crossentropy",
                        required=False)

    parser.add_argument('--verbose',
                        type=int,
                        help='Show more verbose output.',
                        choices=[0, 1, 2],
                        default=1,
                        required=False)

    parser.add_argument('--output_csv',
                        type=str,
                        help='CSV file where save results.',
                        required=False)
    parser.add_argument('--class_attribute',
                        type=str,
                        help='Class to predict.',
                        default="gs_text34",
                        choices=["gs_text34", "module", "site"],
                        required=False)
    return parser


def main():
    """Starting point of the application."""
    args = get_parser().parse_args()
    AutopsiesNeuralNetwork(**args.__dict__).run()


if __name__ == "__main__":
    main()
