###############################################################################
#
# Author: Tyler Teichmann
# Purpose: Helpers for converting RGB values to HSV, and extracting raw data
# Desctiption: This is for the initial data exploration of the images.
# Date: 2025-04-01
#


import os
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import tifffile as tif


def to_hsv(df):
    size = df.shape[0]
    sample = df.sample(1)

    data = df.loc[:, "0":"3071"].to_numpy()
    data = data/255
    data = data.reshape((size, 1024, 3))

    data = mcolors.rgb_to_hsv(data)

    data = data * np.array([360, 100, 100])
    data = data.reshape((size, 3072))

    df.loc[:, "0":"3071"] = data.astype("int64")

    # check_sample(df, sample)

    return df


def check_sample(df, sample):
    rgb = sample.loc[:, "0":"2"].values[0]
    hsv = df.loc[df["id"] == sample["id"].values[0], "0":"2"].values[0]

    test_val = mcolors.rgb_to_hsv(rgb/255)
    test_val = test_val * np.array([360, 100, 100])
    test_val = test_val.astype("int64")

    if np.array_equal(hsv, test_val):
        print("Good Sample")
    else:
        print("Bad Sample")


def extract_rgb():
    training_data = pd.read_csv("train_labels.csv")
    training_data_values = pd.DataFrame()

    for image in training_data["id"]:
        pixels = tif.imread(f"train/{image}.tif")
        pixels = pixels[33:65, 33:65].flatten()
        training_data_values = pd.concat(
            [training_data_values, pd.DataFrame([pixels], dtype=pd.UInt8Dtype())],
            ignore_index=True
        )

    training_data = pd.concat([training_data, training_data_values], axis=1)
    training_data.to_csv("training_data2.csv", index=False)

    print("Training data extracted.")

    ids = [id.removesuffix(".tif") for id in os.listdir("test/")]
    testing_data = pd.DataFrame({"id":ids})
    testing_data_values = pd.DataFrame()

    for image in testing_data["id"]:
        pixels = tif.imread(f"test/{image}.tif")
        pixels = pixels[33:65, 33:65].flatten()
        testing_data_values = pd.concat(
            [testing_data_values, pd.DataFrame([pixels], dtype=pd.UInt8Dtype())],
            ignore_index=True
        )

    testing_data = pd.concat([testing_data, testing_data_values], axis=1)
    testing_data.to_csv("testing_data2.csv", index=False)

    print("Testing data extracted.")


if __name__ == "__main__":
    extract_rgb()