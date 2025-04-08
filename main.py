###############################################################################
#
# Author: Tyler Teichmann
# Purpose: This is my submission for the Histopathologic Cancer Detection 
# Kaggle Challenge.
# Desctiption: Starting with a dataframe that contains image ids and their 
# label, the program reads the center 32x32 pixel RGB values and fits an 
# sklearn Logistic Regression model to that image repeating for each image in
# the training data. The program then reads a test image in the same location,
# assigns a label, and writes the values to submission.csv until all test 
# images are labeled
# Date: 2025-04-01
#


import os
import numpy as np
import pandas as pd
import tifffile as tif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import time


def main(n=None):
    # Initialize the model
    model = LogisticRegression(solver="saga", max_iter=10000)

    # import training id and labels
    training_data = pd.read_csv("train_labels.csv")
    training_data = training_data.sample(n, replace=True)
    X = pd.DataFrame()
    y = training_data["label"]

    start_time = time.time()

    for image in training_data["id"]:
        pixels = tif.imread(f"train/{image}.tif")
        pixels = pixels[33:65, 33:65].flatten()
        X = pd.concat([X, pd.DataFrame([pixels])], ignore_index=True)
        
    end_time = time.time()
    print("Training time: ", (end_time - start_time))

    start_time = time.time()
    
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    model.fit(X, y)

    end_time = time.time()
    print("Fitting time: ", (end_time - start_time))

    ids = [id.removesuffix(".tif") for id in os.listdir("test/")]
    test_data = pd.DataFrame({"id":ids})
    X_hat = pd.DataFrame()
    y_hat = []

    start_time = time.time()

    for image in test_data["id"]:
        pixels = tif.imread(f"test/{image}.tif")
        pixels = pixels[33:65, 33:65].flatten()
        X_hat = pd.concat([X_hat, pd.DataFrame([pixels])], ignore_index=True)

    X_hat = scaler.transform(X_hat)
    y_hat = model.predict(X_hat)

    end_time = time.time()
    print("Testing time: ", (end_time - start_time))

    test_data["label"] = y_hat
    test_data.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main(100000)