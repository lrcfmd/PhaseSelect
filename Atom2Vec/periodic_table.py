ELEMENTS = tuple(
    'H|He|'
    'Li|Be|B|C|N|O|F|Ne|'
    'Na|Mg|Al|Si|P|S|Cl|Ar|'
    'K|Ca|Sc|Ti|V|Cr|Mn|Fe|Co|Ni|Cu|Zn|Ga|Ge|As|Se|Br|Kr|'
    'Rb|Sr|Y|Zr|Nb|Mo|Tc|Ru|Rh|Pd|Ag|Cd|In|Sn|Sb|Te|I|Xe|'
    'Cs|Ba|La|Ce|Pr|Nd|Pm|Sm|Eu|Gd|Tb|Dy|Ho|Er|Tm|Yb|Lu|Hf|Ta|W|Re|Os|Ir|Pt|Au|Hg|Tl|Pb|Bi|Po|At|Rn|'
    'Fr|Ra|Ac|Th|Pa|U|Np|Pu|Am|Cm|Bk|Cf|Es|Fm|Md|No|Lr|Rf|Db|Sg|Bh|Hs|Mt|Ds|Rg'.split('|')
)


LOW_CHARS = "klcganpodeysurtmifhb"
CAP_CHARS = "ZYHGLWKDXRSCTVINFAEUOBMP"
ELE_NUMBERS = len(ELEMENTS)

def lookupEle(element):
    return ELEMENTS.index(element)

if __name__ == "__main__":
    #from Magpie_elemental_features import symbols
    #print(lookupEle("U"))
    print(len(ELEMENTS))
    #for i in ELEMENTS:
    #    if i not in symbols: print(f"Can't find {i}")
    
