import numpy as np
import pandas as pd
import matplotlib as plt
from math import sqrt
from yellowbrick.cluster import KElbowVisualizer
from statistics import stdev

from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
import numpy as np
from scipy.cluster.vq import kmeans,vq
import pandas as pd
from math import sqrt
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

'''Data Prep and Model Evaluation'''
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc, roc_auc_score, mean_squared_error
from keras.utils import to_categorical
from sklearn.metrics import adjusted_rand_score
import random

'''Algos'''
#from kshape.core import kshape, zscore
import tslearn
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import KShape, TimeSeriesScalerMeanVariance
from tslearn.clustering import TimeSeriesKMeans


def strategy_kshape(df, n = 252, K = 4):
    X = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(df.T.values)

    ks = KShape(n_clusters=K, max_iter=100, n_init=100, verbose=0).fit(X)

    index = list(range(len(df.columns)))
    columns = ['strategies', 'clusters', 'selection']

    results = pd.DataFrame(index=index, columns=columns)

    results['clusters'] = ks.labels_
    results['strategies'] = df.columns

    sharpeclusters = []
    df = df.tail(n)
    for i in range(results['clusters'].nunique()):
        l = results.loc[results['clusters'] == i].index.values.astype(int).tolist()
        dfexp = df.pct_change().iloc[:, l].sum(axis=1, skipna=True)
        r = dfexp.mean()
        s = stdev(dfexp)

        sharpeclusters.append(r / s)
    sharpeclusters = np.asanyarray(sharpeclusters)

    selection = sharpeclusters.argmax()
    cond = results['clusters'] == selection

    results['selection'] = np.where(cond, 1, 0)

    return results['strategies'][results['selection'] == 1]

