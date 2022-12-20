import sys
import pandas as pd

df = pd.read_pickle(sys.argv[1])

print(df.head)
