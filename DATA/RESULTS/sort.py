import sys
import pandas as pd

fi = sys.argv[1]
df = pd.read_csv(fi)

df['N'] = df.phases.apply(lambda x: len(x.split()))

b = df[df['N']== 4 ]
b = b[b['average probability Tc>10K']>0.5]
b = b[['phases', 'average probability Tc>10K']]

b.to_csv('quaternary_scon_ICSD.csv', index=False)
