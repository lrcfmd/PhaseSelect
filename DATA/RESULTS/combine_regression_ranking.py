import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

def combine(rank, regress,clas, prop='predicted bg eV'):
    rank = rank.sort_values(['phases'])
    clas = clas.sort_values(['phases'])
    regress = regress.sort_values(['phases'])

    rank = rank[rank['phases'].isin(regress['phases'])]
    clas = clas[clas['phases'].isin(regress['phases'])]

    rank['probability'] = clas['probability'].values
    rank[f'{prop}'] = regress[f'{prop}'].values

    rank = rank.sort_values(by=['probability'])

    # compare probability BG > 4.5 and regression:
    plt.scatter(rank['probability'], rank[f'{prop}'])
    plt.show()

    return rank


def plot_all(df):
    """ """
    #df = df[(df['norm. score'] < 0.1) & (df['probability'] > 0.75)] # mag
    df = df[(df['norm. score'] < 0.18) & (df['probability'] > 0.71)] # scon
    print(df.head(10))

    rank = df['norm. score'].values
    regress = df['predicted Tc'].values
    errors = df['error_bar'].values * 2
    cls = df[''].values
    

    plt.scatter(rank, cls, s=errors, c=regress)
    plt.colorbar(label='Regression, eV')
    #plt.ylabel('Probability of band gap > 4.5 eV')
    #plt.ylabel('Probability of Curie T > 300 eV')
    plt.ylabel('Probability of Tc > 10 K')
    plt.xlabel('Ranking, 1 / Novelty')
    plt.show()


if __name__ == "__main__":
    regression = pd.read_csv(sys.argv[1], comment='#')
    ranking  = pd.read_csv(sys.argv[2])
    clas = pd.read_csv(sys.argv[3], comment='#')

    df = combine(ranking, regression, clas, prop='predicted Tc')
    #df.to_csv('bg_ranking_regression_cls.csv',index=False)
    plot_all(df)
