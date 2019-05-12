"""
IMDB Text Classification
AutoKeras

May 2019

Vocab from [Stanford](http://ai.stanford.edu/~amaas/data/sentiment/)

Preprocessed data from [Keras](https://keras.io/datasets/)
"""
import os
import re
import urllib
import zipfile
import tarfile
import numpy as np
import pandas as pd
import tensorflow as tf

from keras.datasets import imdb
from autokeras.utils import read_tsv_file
from autokeras.text.text_supervised import TextClassifier

def download_url_to_filepath(outpath, url):
    """Create path and download data from url.
    Args
    :outpath: str save to this path
    :url: str url/to/data
    :return: str outpath with data
    """
    fd, fn = re.findall(r"^(.+\/)([^\/]+)$",outpath)[0]
    fp = fd + fn
    if not os.path.exists(fd):
        os.makedirs(fd)
    if not os.path.exists(fp):
        urllib.request.urlretrieve(url, fp) 
    return fp 

def unzip_file(in_path,out_dir):
    """Decompress file (tar.gz or zip)
    Args
    :in_path: str path/to/compressed/file
    :out_dir: str directory with uncompressed folder
    :return: output directory
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        if in_path[-3:]=="zip":
            z = zipfile.ZipFile(in_path,'r')
            z.extractall(out_dir)
            z.close()
        elif in_path[-6:]=="tar.gz":
            tar = tarfile.open(fp)
            tar.extractall(path=out_dir)
            tar.close()    
    return out_dir

def load_data(path):
    """Loads Keras dataset preprocessed into integers by frequency of occurrence, 1 being most frequent
    
    Args
    :path: str path/to/keras/dataset
    :return: tuple(arr,arr)  two tuples of train and test
    """
    (x_train, y_train), (x_test, y_test) = imdb.load_data(path=path)
    return (x_train, y_train), (x_test, y_test)

def convert_labels_to_one_hot(labels, num_labels):
    """One hot encode labels"""
    one_hot = np.zeros((len(labels), num_labels))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot

def convert_int_to_word(integer):
    """Convert processed integer to word from vocab"""
    return vocab.values[integer][0]

def convert_int_to_str_array(array):
    return np.array([" ".join(map(str,[convert_int_to_word(n) for n in integer])) for integer in array])

fp = download_url_to_filepath("/tmp/imdb.tar.gz","http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz")
fd = unzip_file(fp,"/tmp/imdb")
vocab = pd.read_csv(os.path.join(fd,"imdb/aclImdb/imdb.vocab"))

(x_train, y_train), (x_test, y_test) = load_data(path="imdb.npz")
print(f"X,Y Train: {len(x_train),len(x_train[0])},{len(y_train)}")
print(f"X,Y Test: {len(x_test),len(x_test[0])},{len(y_test)}")

x_train, y_train = (x_train, y_train)
x_test, y_test = (x_test, y_test)

x_train = convert_int_to_str_array(x_train)
x_test = convert_int_to_str_array(x_test)

y_train = convert_labels_to_one_hot(y_train, num_labels=2)
y_test = convert_labels_to_one_hot(y_test, num_labels=2)

clf = TextClassifier(verbose=True)
clf.fit(x=x_train, y=y_train, time_limit=12 * 60 * 60)
print("Classification accuracy is : ", 100 * clf.evaluate(x_test, y_test), "%")

