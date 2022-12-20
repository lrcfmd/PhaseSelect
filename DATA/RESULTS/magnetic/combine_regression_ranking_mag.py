import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

def combine(rank, regress,  prop='predicted Tc'):
    rank = rank.sort_values(['phases'])
    regress = regress[regress.phases.isin(rank.phases)]
    rank = rank[rank.phases.isin(regress.phases)]
    regress = regress.sort_values(['phases'])

    rank[f'{prop}'] = regress[f'{prop}'].values

    rank = rank.sort_values(by=['probability'])


    # compare probability BG > 4.5 and regression:
    plt.scatter(rank['probability'], rank[f'{prop}'])
    plt.show()

    return rank


def plot_all(df):
    """ """
    df = df[(df['norm. score'] < 0.1) & (df['probability'] > 0.75)]
    print(df.head(10))

    rank = df['norm. score'].values
    regress = df['predicted Tc'].values
    

    plt.scatter(rank, cls) # s=errors, c=regress)
    plt.colorbar(label='Regression, eV')
    #plt.ylabel('Probability of band gap > 4.5 eV')
    plt.ylabel('Probability of Curie T > 300 eV')
    plt.xlabel('Ranking, 1 / Novelty')
    plt.show()


if __name__ == "__main__":
    regression = pd.read_csv('mag_regressor_candidate_prediction.csv')
    ranking_clas  = pd.read_csv('_mag_P71_R007.csv')

    df = combine(ranking_clas, regression, prop='predicted Tc')
    df.to_csv('mag_ranking_regression_cls.csv',index=False)
    plot_all(df)
