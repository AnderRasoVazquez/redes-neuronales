"""This module handles neural networks."""

from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from time import time, strftime, gmtime

import pandas as pd
import os


class AutopsiesNeuralNetwork(object):
    """Neural network for classification of autopsies."""

    def __init__(self, data_path="files/verbal_autopsies_clean.csv", num_logits=30,
                 num_intermediate=256, num_layers=1, kfold=10, epochs=20,
                 activation_intermediate="sigmoid", activation_output="softmax", optimizer="adam",
                 loss="categorical_crossentropy", verbose=1, output_csv=None):
        self.data_path = data_path
        """Path of the data to read."""
        self.num_logits = num_logits
        """Number of inputs of the neural network."""
        self.num_intermediate = num_intermediate
        """Number of inputs of the neural network intermediate layers."""
        self.num_outputs = None
        """Number of outputs of the neural network. Loading the data sets this variable."""
        self.kfold = kfold
        """Number of folds for the kfold evaluation."""
        self.epochs = epochs
        """Number of iterations of the training for the neural network."""
        self.num_layers = num_layers
        """Number of hidden layers."""
        self.activation_intermediate = activation_intermediate
        """Activation function for hidden layers."""
        self.activation_output = activation_output
        """Activation function for hidden layers."""
        self.optimizer = optimizer
        """Neural network optimizer algorithm."""
        self.loss = loss
        """Loss function."""
        self.verbose = verbose
        """Show more verbose output."""
        self.output_csv = output_csv
        """CSV file where save results."""

    def _load_data(self):
        """Loads dataframe and returns a tuple (data, labels, headers)"""
        df = pd.read_csv(self.data_path)
        df = df.dropna(subset=["open_response", "gs_text34"])  # delete null values from open_response and gs_text34
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df["open_response"])
        y = df["gs_text34"]
        headers = y.unique()
        self.num_outputs = len(headers)
        return X, y, headers

    def _create_model(self):
        """Create neural network model."""
        model = Sequential()
        model.add(Dense(self.num_intermediate, input_shape=(self.num_logits,), activation=self.activation_intermediate))
        for i in range(self.num_layers - 1):
            model.add(Dense(self.num_intermediate, activation=self.activation_intermediate))
        model.add(Dense(self.num_outputs, activation=self.activation_output))
        model.compile(loss=self.loss, optimizer=self.optimizer,  metrics=['accuracy'])
        model.summary()
        return model

    def _sparse_to_pca(self, data):
        """Transforms a sparse matrix to PCA."""
        data = data.todense()
        pca = PCA(n_components=self.num_logits).fit(data)
        data = pca.transform(data)
        return data

    def _save_results(self, header, row):
        """Save neural network results to csv file."""
        file_full_path = os.path.realpath(self.output_csv)
        dir_path = os.path.dirname(file_full_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print("Created: " + dir_path)

        if not os.path.isfile(file_full_path):
            with open(file_full_path, "x") as f:
                f.write(header)
                f.write("\n")
                f.write(row)
                f.write("\n")
            print("Created: " + self.output_csv)
        else:
            with open(file_full_path, "a") as f:
                f.write(row)
                f.write("\n")
            print("Updated: " + self.output_csv)

    def run(self):
        """Run the neural network."""
        now = time()
        data, labels, headers = self._load_data()
        data = self._sparse_to_pca(data)
        labels = pd.get_dummies(labels)  # transform labels to categorical
        estimator = KerasClassifier(build_fn=self._create_model, epochs=self.epochs, verbose=self.verbose)
        kfold = KFold(n_splits=self.kfold, shuffle=True)
        results = cross_val_score(estimator, data, labels, cv=kfold)

        final = time()
        elapsed_time = strftime('%H:%M:%S', gmtime(final - now))
        results_mean = results.mean() * 100
        results_std = results.std() * 100
        csv_header = "time,num_logits,num_intermediate,num_layers,epochs,optimizer, "
        csv_header += "activation_intermediate,activation_output,loss,accuracy,std"
        row = f"{elapsed_time},{self.num_logits},{self.num_intermediate},{self.num_layers},{self.epochs},"
        row += f"{self.optimizer},{self.activation_intermediate},{self.activation_output},{self.loss},"
        row += f"{results_mean:.2f},{results_std:.2f}"

        output = {'COLUMN': csv_header.split(","), 'VALUE': row.split(",")}
        print("\n###### RESULTS ######")
        print(pd.DataFrame(data=output).to_string(index=False))
        print()

        if self.output_csv:
            try:
                print(f"Saving results to {self.output_csv}...")
                self._save_results(header=csv_header, row=row)
            except:
                print("Impossible to save output.")
