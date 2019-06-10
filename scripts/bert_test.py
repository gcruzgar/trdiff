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
    """Load all files from a directory in a DataFrame."""
    data = {}
    data["text"] = []
    for file_path in os.listdir(directory):
        with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
            data["text"].append(f.read())
    
    return pd.DataFrame.from_dict(data)

# Load data
metadata = pd.read_csv("data/timed-un/reliable.dat", sep=' ')
#metadata = remove_outliers(metadata, 'perday', lq=0.05, uq=0.95) # remove outliers

text_path = "data/timed-un/un-readability/"
text_df = load_directory_data(text_path)

