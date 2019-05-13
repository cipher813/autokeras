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
from argparse import ArgumentParser

import cloud.aws.s3.core as s3

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

def prep_dataframe(in_dir,label):
    lst = []
    for fil in os.listdir(in_dir):
        lst.append(open(os.path.join(in_dir,fil),'r').readlines()[0])
    df = pd.DataFrame(lst,columns=["review"])
    df['sentiment']=label
    return df

def prep_sample(in_dir, sample_type):
    neg = prep_dataframe(os.path.join(in_dir,f"aclImdb/{sample_type}/neg/"),0)
    pos = prep_dataframe(os.path.join(in_dir,f"aclImdb/{sample_type}/pos/"),1)
    df = pd.concat([neg,pos])
    df = df.sample(frac=1)
    X = np.array(df['review'])
    Y = np.array(df['sentiment'])
    return X, Y

def convert_labels_to_one_hot(labels, num_labels):
    """One hot encode labels"""
    one_hot = np.zeros((len(labels), num_labels))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--visual-path", help="path to save visualizations"
    )
    args = parser.parse_args()

    # to download required pretrained bert files
    s3.download_dir("s3://nucleus-chc-preprod-datasciences/users/bmcmahon/nas/pytorch_pretrained_bert",
                    os.path.join(os.path.expanduser("~"),".pytorch_pretrained_bert"))

    fp = download_url_to_filepath("/tmp/imdb.tar.gz","http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz")
    fd = unzip_file(fp,"/tmp/imdb")

    x_train, y_train = prep_sample(fd,'train')
    x_test, y_test = prep_sample(fd,'test')

    print(f"X,Y Train: {len(x_train),len(x_train[0])},{len(y_train)}")
    print(f"X,Y Test: {len(x_test),len(x_test[0])},{len(y_test)}")

    y_train = convert_labels_to_one_hot(y_train, num_labels=2)
    y_test = convert_labels_to_one_hot(y_test, num_labels=2)

    # visual_path = args.visual_path if args.visual_path else None
    if not os.path.exists(args.visual_path):
        os.makedirs(args.visual_path)
    clf = TextClassifier(path=args.visual_path,verbose=True)
    clf.fit(x=x_train, y=y_train, time_limit=12 * 60 * 60)
    print("Classification accuracy is : ", 100 * clf.evaluate(x_test, y_test), "%")

    if args.visual_path:
        print(f"Run visualize.py at path {args.visual_path}")
