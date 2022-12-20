import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn import metrics
from scipy.stats import sem

def error(pred, threshold):
    return (pred-threshold)

def plot_error(errors):
    plt.hist(errors, bins=40)
    plt.xlabel('Absolute error, eV')
    plt.ylabel('Number of phase fields')
    plt.show()

def error_vs_value(errors, values):
    errors = [abs(e) for e in errors]
    plt.scatter(values, errors, s=errors, c=errors)
    plt.colorbar(label='Absolute error, |eV|')
    plt.ylabel('Absolute error, |eV|')
    plt.xlabel('Band gap, eV')
    plt.show()

def get_error_for_value(errors, values, a,b):
    band = []
    for e, v in zip(errors, values):
        if a <= v and v <= b:
            band.append(e)
    return band

def average_error_bands(errors, values):
    errors = [abs(e) for e in errors]
    ranges = np.arange(1.0,7.5,0.5)
    ranges = np.arange(0,1500,100)
    ranges = np.arange(0,120,10)
    bands = []
    for i,j in zip(ranges,ranges[1:]):
        band_errors = get_error_for_value(errors, values, i,j)
        bands.append(band_errors)
    return bands, ranges[:-1]

def r2c(data, threshold, name='band gap eV':
    """ Take a regressor scores and calculate classification accuracy """

    y_test = data[f'{name}']
    y_predict = data[f'{name}']

    errors = [error(i,j) for i,j in zip(y_predict, y_test)]

    #plot_error(erros)

    #error_vs_value(errors, y_test)

    bands, ranges = average_error_bands(errors, y_test)

    #plt.boxplot(bands, positions=[i+0.25 for i in ranges], widths=0.3)
    plt.boxplot(bands, positions=ranges,  widths=5)
    #plt.boxplot(bands, positions=ranges,  widths=40)
    #plt.xticks(list(ranges)+[7.0], [i for i in ranges]+[7.0])
    #plt.xlabel('Curie T, K', fontsize=14)
    plt.xlabel(r'T$_{c}$, K', fontsize=14)
    plt.ylabel('Absolute error, |K|', fontsize=14)
    plt.show()

def add_errors(datafile, testfile):
    """ Take errors from validation set and add error bars to predicted data """

    data = pd.read_csv(datafile)
    test = pd.read_csv(testfile)

    y_test = data['true bg eV']
    y_predict = data['predicted bg eV']
    errors = [error(i,j) for i,j in zip(y_predict, y_test)]
    bands, ranges = average_error_bands(errors, y_test)
    bands_max = [max(b) - min(b) for b in bands]
    bands_max = [np.average(b) for b in bands]
    print(bands_max)

    predictions = list(test['predicted bg eV'])
    error_bars = np.where((predictions < min(ranges)) & (predictions >= max(ranges)), max(bands_max), 0)
    print( len(error_bars[error_bars != 0]))

    ranges = [ (a,b) for a,b in zip(ranges, ranges[1:])]
    print(ranges)

    # create error bars based on the errors in validation set
    for i,t in enumerate(ranges):
        a,b = t
        for j, value in enumerate(predictions):
            if a <= value and value < b:
                error_bars[j] = bands_max[i]

    test['error_bar'] = error_bars
    test = test[['phases', 'predicted bg eV', 'error_bar']]
    test = test.sort_values(by=['predicted bg eV'])

    test.to_csv(testfile, index=False)

def classify(data, threshold=10):
    y_test = data['true Tc']
    y_predict = data['predicted Tc']
   #y_test = data['true bg eV']
   #y_predict = data['predicted bg eV']
    
    y_test = np.where(y_test>=threshold, 1, 0)
    y_predict = np.where(y_predict>=threshold, 1, 0)

    acc = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for t,p in zip(y_test, y_predict):
        if t == p:
            acc += 1
            if t:
                tn += 1
            else:
                tp += 1
        elif t:
            fn += 1
        else:
            fp += 1


    print('Accuracy', acc/len(y_test))
    print('TP', tp/len(y_test))
    print('TN', tn/len(y_test))
    print('F1', tp / (tp + 0.5* (fp + fn)))


if __name__ == "__main__":
    # for plotting:
   df = pd.read_csv(sys.argv[1])
   r2c(df, 4.5)
   #classify(df, 300)
   #residuals(df)

   #train = sys.argv[1]
   #unexplored = sys.argv[2]
   #add_errors(train, unexplored)
