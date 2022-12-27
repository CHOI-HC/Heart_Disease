#%%
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report
from sklearn.metrics import *


#%%
def playML(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    #classification report
    train_report = classification_report(y_train, y_pred_train)
    test_report = classification_report(y_test, y_pred_test)
    print(train_report)
    print(test_report)
    #roc_score
    roc_score_train = roc_auc_score(y_train, y_pred_train)
    roc_score_test = roc_auc_score(y_test, y_pred_test)
    print(f"######{model.__class__.__name__}######")
    print(f"roc_score_train :", roc_score_train)
    print(f"roc_score_test :", roc_score_test)
    print('='*80)
    return y_pred_test

def roc_curve_plot(model, X_train, X_test, y_train, y_test):
    y_pred_test = playML(model, X_train, X_test, y_train, y_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)
    plt.plot(fpr, tpr, label=f'{model.__class__.__name__} ROC curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random', color='red')
    plt.xlabel('FPR( 1 - Sensitivity )')
    plt.ylabel('TPR( Recall )')
    plt.legend()

def learning_curve_plot(model, X_train, y_train, epochs, cv):
    #Learning Curve 그리기
    train_sizes, train_score, test_score = learning_curve(model, X_train, y_train, train_sizes=np.linspace(.1, 1.0, epochs), cv=cv)
    train_mean = np.mean(train_score, axis=1)
    test_mean = np.mean(test_score, axis=1)

    plt.plot(train_sizes, train_mean, "-o", label="train")
    plt.plot(train_sizes, test_mean, "-o", label="test")
    plt.title(f"{model.__class__.__name__} Learning Curve", size=20)
    plt.xlabel("Train Sizes", fontsize=15)
    plt.ylabel("Score", fontsize=15)
    plt.legend()
# %%
