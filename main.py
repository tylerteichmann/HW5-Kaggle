###############################################################################
#
# Author: Tyler Teichmann
# Purpose: This is my submission for the Histopathologic Cancer Detection 
# Kaggle Challenge.
# Desctiption: Starting with a dataframe that contains image ids and their 
# label, the program reads the center 32x32 pixel RGB values and fits an 
# sklearn Logistic Regression or MLP model to that image repeating for each image in
# the training data. The program then reads a test image in the same location,
# assigns a label, and writes the values to submission.csv until all test 
# images are labeled
# Date: 2025-04-01
#


import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import time
import helpers


def main():
    training_file = "train/training_data.csv"
    testing_file = "test/testing_data.csv"

    data_exists = (
        os.path.exists(f"./{training_file}") and
        os.path.exists(f"./{testing_file}")
    )

    training_data = pd.read_csv(training_file)
    testing_data = pd.read_csv(testing_file)

    if not data_exists:
        print("Extracting data, please wait...")
        helpers.extract_rgb()

    label_data("logistic", training_data, testing_data, "rgb", 22000)
    label_data("mlp", training_data, testing_data, "rgb", 22000)
    label_data("logistic", training_data, testing_data, "hsv", 22000)
    label_data("mlp", training_data, testing_data, "hsv", 22000)

    label_data("logistic", training_data, testing_data, "rgb")
    label_data("mlp", training_data, testing_data, "rgb")
    label_data("logistic", training_data, testing_data, "hsv")
    label_data("mlp", training_data, testing_data, "hsv")


def label_data(model_type, training_data, testing_data, feature="rgb", samples=None):

    # Initialize the model
    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "mlp":
        model = MLPClassifier(max_iter=1000)
    else:
        print("Invalid Selection")
        return


    # Extract features and labels
    _, X, y = get_xy(training_data, samples)
    test_ids, X_hat, y_hat = get_xy(testing_data)


    # Convert to hsv if selected
    if feature == "hsv":
        X = helpers.to_hsv(X)
        X_hat = helpers.to_hsv(X_hat)


    # Scale the features
    scaler = StandardScaler().fit(X)

    X = scaler.transform(X)
    X_hat = scaler.transform(X_hat)


    # Fit the model with training data
    print(f"Fitting {model_type} model using {X.shape[0]} samples and {X.shape[1]} features")
    start_time = time.time()
    model.fit(X, y)
    end_time = time.time()
    print("Fitting time: ", (end_time - start_time))


    # Predict test data labels
    print(f"Testing {model_type} model")
    start_time = time.time()
    y_hat = model.predict(X_hat)
    end_time = time.time()
    print("Testing time: ", (end_time - start_time))


    # Write results to csv file
    print("Exporting...")
    submission = pd.DataFrame({"id":test_ids, "label":y_hat})
    submission.to_csv(f"submission_{model_type}_{feature}_{X.shape[0]}.csv", index=False)
    print("Done")


def get_xy(data, samples=None):

    if not samples == None:
        df = data.sample(samples, replace=True)
    else:
        df = data

    ids = df["id"]
    X = df.loc[:,"0":"3071"]

    if "label" in df.columns:
        y = df["label"]
    else:
        y = []

    return (ids, X, y)


if __name__ == "__main__":
    main()