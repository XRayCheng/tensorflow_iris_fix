from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import itertools
import pandas as pd

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"
IRIS_PREDICT = "iris_predict.csv"


tf.logging.set_verbosity(tf.logging.INFO)

COLUMNS = ["SepalLength", "SepalWidth", "PetalLength", "PetatlWidth", "Species"]
FEATURES = ["SepalLength", "SepalWidth", "PetalLength", "PetatlWidth"]
LABEL = "Species"

def input_fn(data_set):
  feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
  labels = tf.constant(data_set[LABEL].values)
  return feature_cols, labels


def main(unused_argv):
# Load datasets.

# Specify that all features have real-value data
#feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

    training_set=pd.read_csv(IRIS_TRAINING, skipinitialspace=True,skiprows=1, names=COLUMNS)
    test_set = pd.read_csv(IRIS_TEST, skipinitialspace=True,
                         skiprows=1, names=COLUMNS)

    prediction_set = pd.read_csv(IRIS_PREDICT, skipinitialspace=True,
                         skiprows=1, names=COLUMNS)


    # Feature cols
    feature_cols = [tf.contrib.layers.real_valued_column(k)
                  for k in FEATURES]

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_cols,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="/tmp/iris_model")

    # Fit model.
    classifier.fit(input_fn=lambda: input_fn(training_set), steps=2000)


    # Evaluate accuracy.
    #accuracy_score = classifier.evaluate(x=test_set.data,
    #                                     y=test_set.target)["accuracy"]
    # Score accuracy
    accuracy_score = classifier.evaluate(input_fn=lambda: input_fn(test_set), steps=1)
    print('Accuracy: {}'.format(accuracy_score))

    # Classify two new flower samples.
    y = classifier.predict(input_fn=lambda: input_fn(prediction_set))
    predictions = list(itertools.islice(y, 2))
    print("Predictions: {}".format(str(predictions)))

if __name__ == "__main__":
  tf.app.run()
