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
    print(df.shape)

    rank = df['norm. score'].values
    cls = df['probability'].values
    regress = df['predicted Tc'].values
    

    plt.scatter(rank, cls, c=regress) # s=errors, c=regress)
    plt.colorbar(label='Regression, K')
    plt.ylabel('Probability of Tc > 10 K')
    plt.xlabel('Ranking, 1 / Novelty')
    plt.show()


if __name__ == "__main__":
    regression = pd.read_csv('scon_regressor_candidate_prediction.csv')
    ranking_clas  = pd.read_csv('_scon_P71_R18.csv')
    df = combine(ranking_clas, regression)

    df.to_csv('scon_ranking_regression_cls.csv',index=False)
    plot_all(df)
