"""This module handles neural networks."""

from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report

import pandas as pd
import numpy as np
import os


def report_average(*args):
    """Uses the reports of each k fold to create an average report."""
    data = {}
    for report in args:

        report = report.split("\n")
        report = list(filter(None, report))
        header = report[0]
        header = list(filter(None, header.split(" ")))

        body = report[1:]
        for line in body:
            line = list(filter(None, line.split("  ")))
            numbers = [float(i) for i in line[1:]]
            numbers = np.asarray(numbers)
            index = line[0].strip()
            if index in data:
                data[index].append(numbers)
            else:
                data[index] = [numbers]

    for key, value in data.items():
        value = np.asarray(value)
        value = np.average(value, axis=0)
        data[key] = value

    df = pd.DataFrame(data, index=header)
    df = df.T  # transpose
    df = df.round(2)  # two decimals
    averages = ['micro avg', 'macro avg', 'weighted avg']
    final_index = [i for i in df.index if i not in averages]
    final_index += averages
    df = df.reindex(final_index)
    df = df.drop(columns=["support"])

    return df


class AutopsiesNeuralNetwork(object):
    """Neural network for classification of autopsies."""

    def __init__(self, data_path="files/verbal_autopsies_clean.csv", num_logits=30,
                 num_intermediate=256, num_layers=1, kfold=10, epochs=20,
                 activation_intermediate="sigmoid", activation_output="softmax", optimizer="adam",
                 loss="categorical_crossentropy", verbose=1, output_file=None, class_attribute="gs_text34", **kwargs):
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
        self.output_file = output_file
        """File where save results."""
        self.class_attribute = class_attribute
        """Class attribute to predict."""

    def _load_data(self):
        """Loads dataframe and returns a tuple (data, labels, headers)"""
        df = pd.read_csv(self.data_path)
        df = df.dropna(subset=["open_response", "gs_text34"])  # delete null values from open_response and gs_text34
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df["open_response"])
        print(X.shape)
        y = df[self.class_attribute]
        print(y.shape)
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

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=["accuracy", self.loss])
        model.summary()
        return model

    def _sparse_to_pca(self, data):
        """Transforms a sparse matrix to PCA."""
        data = data.todense()
        pca = PCA(n_components=self.num_logits).fit(data)
        data = pca.transform(data)
        return data

    def _save_results(self, results):
        """Save neural network results to csv file."""
        file_full_path = os.path.realpath(self.output_file)
        dir_path = os.path.dirname(file_full_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print("Created: " + dir_path)

        with open(file_full_path, "w") as f:
            f.write(results)
            print("Created: " + self.output_file)

    def run(self):
        """Run the neural network."""
        data, labels, headers = self._load_data()
        data = self._sparse_to_pca(data)
        labels = pd.get_dummies(labels, dummy_na=False).astype(np.float32)  # transform labels to categorical
        kfold = KFold(n_splits=self.kfold, shuffle=True)

        classification_reports = []

        estimator = KerasClassifier(build_fn=self._create_model, epochs=self.epochs, verbose=self.verbose)
        i = 0
        for train_index, test_index in kfold.split(data):
            i += 1
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
            # estimator = None
            # estimator = KerasClassifier(build_fn=self._create_model, epochs=self.epochs, verbose=self.verbose)
            history = estimator.fit(X_train, y_train, epochs=self.epochs, validation_data=(X_test, y_test))

            y_pred = estimator.predict(X_test)
            y_pred = [headers[i] for i in y_pred]

            y_test = y_test.idxmax(axis=1)
            y_test_class = y_test
            y_pred_class = y_pred

            report = classification_report(y_test_class, y_pred_class, labels=np.unique(y_pred))
            # cm = confusion_matrix(y_test_class, y_pred_class)
            classification_reports.append(report)

        df_results = report_average(*classification_reports)
        df_results = df_results.to_string()

        result = f"\n#### AVERAGE REPORT FROM {self.kfold} FOLD CROSS VALIDATION  ####\n"
        result += df_results
        print(result)

        if self.output_file:
            try:
                print(f"Saving results to {self.output_file}...")
                self._save_results(result)
            except:
                print("Impossible to save output.")

        return history


