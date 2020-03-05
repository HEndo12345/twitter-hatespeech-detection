import pickle as pkl

def save_pkl(dest, data):
    with open(dest, 'wb') as f:
        pkl.dump(data, f)
    return
  
def load_pkl(dest):
    with open(dest, 'rb') as f:
        data = pkl.load(f)
    return data

import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

def print_cmx(y_true, y_pred):
    cmx_data = confusion_matrix(y_true, y_pred)
    
    df_cmx = pd.DataFrame(cmx_data, index=['hate','offend','neither'], columns=['hate','offend','neither'])
    df_cmx = df_cmx.div(df_cmx.sum(axis=1), axis=0)*100
    
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cmx, annot=True)
    plt.show()
    return df_cmx

import numpy as np
def print_topk(feat_name, clf, class_labels, k):
    """Prints features with the highest coefficient values, per class"""
    feature_names = feat_name
    for i, class_label in enumerate(class_labels):
        top10 = np.argsort(clf.coef_[i])[-k:]
        print("%s: %s" % (class_label,
              " ".join(feature_names[j] for j in top10)))

import random
def shuffle_data(x, y):
    sets = list(zip(x,y))
    random.shuffle(sets)
    return zip(*sets)

import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def print_cmx(y_true, y_pred, color = 'Oranges'):
    cmx_data = confusion_matrix(y_true, y_pred)
    
    df_cmx = pd.DataFrame(cmx_data, index=['hate','offenive','neither'], columns=['hate','offensive','neither'])
    df_cmx = df_cmx.div(df_cmx.sum(axis=1), axis=0)*100
    
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cmx, annot=True, cmap=color)
    plt.show()
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))
    
    return df_cmx

def tokenize(text):
    return text.split()