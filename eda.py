###############################################################################
#
# Author: Tyler Teichmann
# Purpose: This is my submission for the Histopathologic Cancer Detection 
# Kaggle Challenge.
# Desctiption: This is for the initial data exploration of the images. There
# are three idioms used for analysis. 1. A colormesh to isolate different 
# values from two differently labeled images. 2. A histogram to analyze
# distribution of mean color values. 3. A scatterplot to compare distribution
# against two of three values.
# Date: 2025-04-01
#


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import helpers


def main(samples=None, hsv=False):
    # Read in the test labels
    file = "training_data.csv"
    df = pd.read_csv(file)
    
    # Randomly sample the data
    if not samples == None:
        # Select random non-cancerous ids
        non_cancerous_ids = df.loc[df["label"] == 0, "id"].sample(samples, replace=True)
        non_cancerous_ids = non_cancerous_ids.to_numpy()

        # Select random cancerous ids
        cancerous_ids = df.loc[df["label"] == 1, "id"].sample(samples, replace=True)
        cancerous_ids = cancerous_ids.to_numpy()

        # Extract RGB data
        df = pd.concat([
            df.loc[df["id"].isin(non_cancerous_ids)],
            df.loc[df["id"].isin(cancerous_ids)]
        ])

    if hsv:
        df = helpers.to_hsv(df)

    if df.shape[0] == 2:
        plot_one(df, hsv)
    else:
        plot_values(df, hsv)
        plot_many(df, hsv)



def plot_one(df, hsv=False):
    v1_columns = [str((3 * i) + 0) for i in range(1024)]
    v2_columns = [str((3 * i) + 1) for i in range(1024)]
    v3_columns = [str((3 * i) + 2) for i in range(1024)]

    if hsv:
        c1, c2, c3 = "hsv", "binary", "binary"
        t1, t2, t3 = "Pixel Hue", "Pixel Saturation", "Pixel Value"
    else:
        c1, c2, c3 = "Reds_r", "Greens_r", "Blues_r"
        t1, t2, t3 = "Pixel Red Value", "Pixel Green Value", "Pixel Blue Value"

    fig, axs = plt.subplots(nrows=2, ncols=4)

    for i in range(2):
        base = df.loc[df["label"] == i, "0":"3071"].to_numpy()
        v1 = df.loc[df["label"] == i, v1_columns].to_numpy()
        v2 = df.loc[df["label"] == i, v2_columns].to_numpy()
        v3 = df.loc[df["label"] == i, v3_columns].to_numpy()

        base = base.reshape(32, 32, 3)
        v1 = v1.reshape((32, 32))
        v2 = v2.reshape((32, 32))
        v3 = v3.reshape((32, 32))

        if hsv:
            base = base / np.array([360, 100, 100])
            base = mcolors.hsv_to_rgb(base)
        else:
            base = base.astype("uint8")

        axs[i, 0].pcolormesh(v1, cmap=c1)
        axs[i, 1].pcolormesh(v2, cmap=c2)
        axs[i, 2].pcolormesh(v3, cmap=c3)
        axs[i, 3].pcolormesh(base)

    axs[0, 0].set_title(t1)
    axs[0, 1].set_title(t2)
    axs[0, 2].set_title(t3)
    axs[0, 3].set_title("Base Image")

    plt.show()


def plot_many(df, hsv=False):
    transparency = 0.5

    v1_columns = [str((3 * i) + 0) for i in range(1024)]
    v2_columns = [str((3 * i) + 1) for i in range(1024)]
    v3_columns = [str((3 * i) + 2) for i in range(1024)]

    if hsv:
        # c1, c2, c3 = "hsv", "binary", "binary"
        t1, t2, t3 = "Image Mean Pixel Hue", "Image Mean Pixel Saturation", "Image Mean Pixel Value"
    else:
        # c1, c2, c3 = "Reds_r", "Greens_r", "Blues_r"
        t1, t2, t3 = "Image Mean Red Value", "Image Mean Green Value", "Image Mean Blue Value"

    fig, axs = plt.subplots(nrows=1, ncols=3)

    print(f"Sample size {df.shape[0]}")

    for i in range(2):
        if i == 0:
            label = "Non-Cancerous"
            color = "green"
        else:
            label = "Cancerous"
            color = "red"

        v1 = df.loc[df["label"] == i, v1_columns].mean(axis=1)
        v2 = df.loc[df["label"] == i, v2_columns].mean(axis=1)
        v3 = df.loc[df["label"] == i, v3_columns].mean(axis=1)

        axs[0].hist(v1, bins=32, alpha=transparency, label=label, color=color)
        axs[1].hist(v2, bins=32, alpha=transparency, label=label, color=color)
        axs[2].hist(v3, bins=32, alpha=transparency, label=label, color=color)

    axs[0].set_title(t1)
    axs[1].set_title(t2)
    axs[2].set_title(t3)
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    plt.show()


def plot_values(df, hsv):
    transparency = 0.5

    v1_columns = [str((3 * i) + 0) for i in range(1024)]
    v2_columns = [str((3 * i) + 1) for i in range(1024)]
    v3_columns = [str((3 * i) + 2) for i in range(1024)]

    if hsv:
        # c1, c2, c3 = "hsv", "binary", "binary"
        t1, t2, t3 = "Image Mean HS", "Image Mean HV", "Image Mean SV"
    else:
        # c1, c2, c3 = "Reds_r", "Greens_r", "Blues_r"
        t1, t2, t3 = "Image Mean RG", "Image Mean RB", "Image Mean BG"

    fig, axs = plt.subplots(nrows=1, ncols=3)

    print(f"Sample size {df.shape[0]}")

    for i in range(2):
        if i == 0:
            label = "Non-Cancerous"
            color = "green"
        else:
            label = "Cancerous"
            color = "red"

        v1 = df.loc[df["label"] == i, v1_columns].mean(axis=1)
        v2 = df.loc[df["label"] == i, v2_columns].mean(axis=1)
        v3 = df.loc[df["label"] == i, v3_columns].mean(axis=1)

        axs[0].scatter(v1, v2, s=0.1, alpha=transparency, label=label, color=color)
        axs[1].scatter(v1, v3, s=0.1, alpha=transparency, label=label, color=color)
        axs[2].scatter(v2, v3, s=0.1, alpha=transparency, label=label, color=color)

    axs[0].set_title(t1)
    axs[1].set_title(t2)
    axs[2].set_title(t3)
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    plt.show()



if __name__ == "__main__":
    main(hsv=False)