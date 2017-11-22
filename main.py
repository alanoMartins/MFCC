import os.path
import pandas as pd
from classifier import Classifier
import numpy as np

from extractor import Extractor

SAMPLE_DIR = 'samples/'


def save(features):
    df = pd.DataFrame(features)
    df.to_csv('output/feature.csv')

def read():
    return pd.read_csv('output/feature.csv')

def build_row(path):
    #result = 1 if 'alano' in path else 0
    result = -1
    if 's1' in path:
        result = 0
    if 's2' in path:
        result = 1
    if 's3' in path:
        result = 2

    #arr = extractor.mfcc(path).flatten()
    arr = extractor.feature_lib(path).flatten()
    arr = np.insert(arr, 0, result, axis=0)
    return arr


if __name__ == '__main__':

    extractor = Extractor('sanderson')

    files = next(os.walk(SAMPLE_DIR))[2]
    files = list(files)
    features = (build_row(SAMPLE_DIR + file) for file in files)
    features = list(features)
    save(features)

    # c = Classifier()
    # c.knn(read())
