import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix,ConfusionMatrixDisplay, roc_curve, roc_auc_score, matthews_corrcoef
import sys
import re

def mcc(df, Tc=10):
    """ Matthews' coefficients """
    y_true = np.where(df['max Tc'].values > Tc, 1, 0)
    y_pred = np.where(df['probability_1'].values > 0.5, 1, 0) 
    sample_weight = len(y_true[y_true == 1]) / len(y_true)
    mc = matthews_corrcoef(y_true, y_pred)
    print("Matthews' coefficients:", mc)
    return mc


def roc(df, Tc):
    print(df.head())
    y_true = np.where(df['max Tc'].values > Tc, 1, 0)
    y_pred = df['probability_1'].values
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr)
    plt.title('ROC curve')
    plt.xlabel(f'False positive rate (false Tc>{Tc}K)')
    plt.ylabel(f'True positive rate (true Tc>{Tc}K)')
    plt.show()
    print('AUC', roc_auc_score(y_true, y_pred))

def remove_FP(df, Tc, prob, ffile):
    x_true = np.where(df['max Tc'].values > Tc, 1, 0)
    x_pred = np.where((df['probability_1'].values > prob), 1, 0)
    with open(ffile, 'a') as ff:
        print('The accuracy of the model is:', accuracy_score(x_true, x_pred), file=ff)
        print('The f1 score of the model is:',f1_score(x_true, x_pred), file=ff)
        print(f"Data distribution: 0 (<{Tc}): {len(x_true) - sum(x_true)}; 1(>{Tc}):{ sum(x_true)}", file=ff)

    labels = [fr"T$_c < {Tc}$K", fr"T$_c >= {Tc}$K"]

    cm = confusion_matrix(x_true, x_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig = plt.figure()
    disp.plot(cmap='Blues')
    plt.show()
    disp.figure_.savefig(ffile+'.png', dpi=300, pad_inches=1)

def true_high(df, prob, Tc):
    """ Determine true high """
    true = df[df['probability_1'] > prob]
    h_true = np.where(true['max Tc'].values > Tc, 1, 0)
    x_pred = true['prediction']
    acc = sum(h_true) / sum(x_pred)
    print(f"with probability {prob} accuracy: {acc}")
    return acc

def plot_confusion(df, Tc, ffile):
    """ Plot cunfusion matrix """

    # get true and pred:
    x_true = np.where(df['max Tc'].values > Tc, 1, 0)
    x_pred = df['prediction'].values

    with open(ffile, 'a') as ff:
        print('The accuracy of the model is:', accuracy_score(x_true, x_pred), file=ff)
        print('The f1 score of the model is:',f1_score(x_true, x_pred), file=ff)
        print(f"Data distribution: 0 (<{Tc}): {len(x_true) - sum(x_true)}; 1(>{Tc}):{ sum(x_true)}", file=ff)

    labels = [fr"T$_c < {Tc}$K", fr"T$_c >= {Tc}$K"]

    cm = confusion_matrix(x_true, x_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig = plt.figure()
    disp.plot(cmap='Blues')
    plt.show()
    disp.figure_.savefig(ffile+'.png', dpi=300, pad_inches=1)

    return accuracy_score(x_true, x_pred), f1_score(x_true, x_pred)

def reduce_duplicate(df):
    phases = [' '.join(sorted(x.split())) for x in df['phases'].values]
    df = df.assign(phases=phases)
    print(df.shape)
    dt =  df.drop_duplicates(subset=['phases'])
    print(dt.shape)
    return dt

if __name__ == "__main__":
    ffile = sys.argv[1]
    df = pd.read_pickle(ffile)
    #df = reduce_duplicate(df)
    #plot_confusion(df, 10, 'Precision_model.txt')
    #p = 0.80
    #remove_FP(df, 10, p, f'Precision_model_prob_{p}.txt')
  
    # get True high threshold:
    #for p in np.arange(0.5, 1.0, 0.05):
    #     true_high(df, p, 10)

    #roc(df, Tc=300)

    mc = mcc(df)
