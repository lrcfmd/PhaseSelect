import sys
import pandas as pd

def get_number(x):
    return len(x.split())

def get_ternary(df):
   df['length'] = list(map(get_number, df.phases))
   ternary = df[df['length'] == 3]
   print(len(ternary.phases))


if __name__ == '__main__':
    try:
        fname = sys.argv[1]
    except Exception:
        print("Give a file name")

    #df = pd.read_csv(fname)
    df = pd.read_pickle(fname)
    get_ternary(df)
