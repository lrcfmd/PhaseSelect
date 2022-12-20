import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler

features = 'Number, AtomicVolume, AtomicWeight, GSestBCClatcnt, Column, CovalentRadius,Electronegativity, FirstIonizationEnergy, GSbandgap, GSenergy_pa, GSestBCClatcnt, GSestFCClatcnt, GSmagmom, GSvolume_pa, ICSDVolume, MendeleevNumber, NdUnfilled, NdValence, NfUnfilled, NfValence, NpUnfilled, NpValence, NsUnfilled, NsValence, NUnfilled, NValance, Polarizability, Row'.split(',')
features = [i.strip() for i in features]


def read_features(f):
    lines = open(f,'r').readlines()
    return [float(l.strip()) if l.strip().isdigit else 0 for l in lines]

def sym2num(element, features, scale=False):
    """create n new descriptors from the tables"""
    n = len(features)
    dics = [ {} for i in range(n)]

    for i in range(n):
        try:
            table = read_features(f'magpie_tables/{features[i]}.table')
            dics[i]  = {sym: float(num) for sym, num in zip(symbols, table)}
        except:
            pass

    elemental_features = []
    for i in range(n): 
        elemental_features.append(float(dics[i][element]))

    X = np.array(elemental_features)
    
    if scale:
        X = MaxAbsScaler().fit_transform(X) 

    return list(X)

def num2sym(number, feature):
    numbers = [str(int(num)) for num in read_features(f'magpie_tables/{feature}.table')]
    dic = {num: sym for num, sym in zip(numbers, symbols)}
    return dic[str(int(number))]

if __name__ == "__main__":
    try:
        symbols = [s.strip() for s in open('Atom2Vec/magpie_tables/Abbreviation.table', 'r').readlines()]
    except:
        symbols = [s.strip() for s in open('magpie_tables/Abbreviation.table', 'r').readlines()]

    #X = sym2num('Cl', ['Number', 'AtomicVolume', 'AtomicWeight'], scale=False)
    #features = [i.strip() for i in open('list_tables', 'r').readlines()]
    
    dic = {symbol: sym2num(symbol, features, scale=False) for symbol in symbols}
    
    df = pd.DataFrame(dic)
    print(df.shape)
    df.to_pickle('magpie_atomic_features.pkl')
