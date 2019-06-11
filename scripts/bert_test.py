#!/usr/bin/env python3

import os
import re

from scripts.utils import remove_outliers

import numpy as np
import pandas as pd 
import tensorflow as tf 
import tensorflow_hub as hub 
from datetime import datetime
from sklearn.model_selection import train_test_split

import bert
from bert import tokenization
from bert import optimization
from bert import tokenization

def load_directory_data(directory):
    """Load all files from a directory into a DataFrame."""

    data = pd.DataFrame()
    data["filename"] = []
    data["text"] = []
    i=0
    for file_path in os.listdir(directory):

        with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
            data.loc[i] = [file_path, f.read()]

        i+=1

    return data

# Load data
metadata = pd.read_csv("data/timed-un/reliable.dat", sep=' ')

text_path = "data/timed-un/un-readability/"
text_df = load_directory_data(text_path)
text_df['filename'] = text_df['filename'].str.lower() # change filenames to lower case to match metadata

# Join all into single dataframe
df = metadata.merge(text_df, left_on="ID", right_on="filename")
df.drop(columns=["filename"], inplace=True)

# Remove outliers
df = remove_outliers(df, 'perday', lq=0.05, uq=0.95) 
