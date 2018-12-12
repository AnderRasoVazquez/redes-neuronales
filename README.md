# Parameters

This applications creates a custom neural network to do text classification on verbal autopsies.

```
usage: __main__.py [-h] --data_path DATA_PATH [--num_logits NUM_LOGITS]
                   [--num_intermediate NUM_INTERMEDIATE]
                   [--num_layers NUM_LAYERS] [--kfold KFOLD] [--epochs EPOCHS]
                   [--activation_intermediate {sigmoid,relu,softmax}]
                   [--activation_output {sigmoid,relu,softmax}]
                   [--optimizer {adam,sgd}]
                   [--loss {categorical_crossentropy,binary_crossentropy}]
                   [--verbose {0,1,2}] [--output_file OUTPUT_FILE]
                   [--class_attribute {gs_text34,module,site}]
                   [--plot_path PLOT_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path of the data to read.
  --num_logits NUM_LOGITS
                        Number of inputs of the neural network. >=1
  --num_intermediate NUM_INTERMEDIATE
                        Number of inputs of the neural network intermediate
                        layers. >=1
  --num_layers NUM_LAYERS
                        Number of hidden layers. >=1
  --kfold KFOLD         Number of folds for the kfold evaluation. >=2
  --epochs EPOCHS       Number of iterations of the training for the neural
                        network. >=1
  --activation_intermediate {sigmoid,relu,softmax}
                        Activation function for hidden layers.
  --activation_output {sigmoid,relu,softmax}
                        Activation function for hidden layers.
  --optimizer {adam,sgd}
                        Neural network optimizer algorithm.
  --loss {categorical_crossentropy,binary_crossentropy}
                        Loss function.
  --verbose {0,1,2}     Show more verbose output.
  --output_file OUTPUT_FILE
                        File where save results.
  --class_attribute {gs_text34,module,site}
                        Class to predict.
  --plot_path PLOT_PATH
                        Path to save plot.


```
## EXAMPLES

```
$ python3 __main__.py  --data_path files/verbal_autopsies_clean.csv --epochs 20 --kfold 10 --num_logits 3 --num_intermediate 64 --num_layers 1 --class_attribute module
$ python3 __main__.py  --data_path files/verbal_autopsies_clean.csv --epochs 100 --kfold 10 --num_logits 50 --num_intermediate 128 --num_layers 2 --class_attribute gs_text34 --output_file result_gs_text34_base.txt --plot_path plot_gs_text34_base.png
```

