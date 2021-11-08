import numpy as np
from AE import train_AE
from periodic_table import ELEMENTS
from Magpie_elemental_features import *
from EnvMatrix import EnvsMat

class Atom2Vec:
    def __init__(self, filename, k, atomvec_file="Atom2Vec/atoms_AE_vec_onehot.txt", mode='classify'):
        """ Reads environments and exctracts atomic features

         Parameters:
         ----------
         mode: str;'classify' (default): returns raw environments
                   'rank': reads precalculated atomic vectors 
                   'AE':  runs AE to extract atomic vectors
                   'magpie' - use Magpie features instead of environments 

         filename: str; file with structural environments  
 
         """

        self.atoms_vec, self.atoms_index = self.read_vecs(atomvec_file)
        self.elements = [ELEMENTS[i] for i in self.atoms_index]
        envs_mat = EnvsMat(filename)
        self.envs_mat = envs_mat.envs_mat

        if mode in ['classify','rank', 'AE']:
            self.atoms_index = envs_mat.atoms

            if mode == 'AE':
                self.generateVec_AE(self.envs_mat, k)

        elif mode == 'magpie':
            self.envs_mat = self.magpie(features)
            print(f"Magpie features, matrix shape: {self.envs_mat.shape}")
            self.generateVec_AE(self.envs_mat, k)

    def augment_features(self, envs_mat, features):
        """ Add Magpie features to each element in envs_mat """
        print(f"Augmenting raw matrix of elemental descriptors with Magpie features. Mat shape: {envs_mat.shape}")
        
        new_mat = np.zeros((len(self.elements), len(envs_mat[0]) + len(features)))
        new_mat[:,:len(envs_mat[0])] = envs_mat
      
        for i, el in enumerate(self.elements):
            new_mat[i,-len(features):] = sym2num(el, features)

        return new_mat

    def magpie(self, features):
        """ Create a matrix of Magpie elemental features """
        mat = np.zeros((len(self.elements), len(features)))
        for i, el in enumerate(self.elements):
            mat[i,:] = sym2num(el, features)

        return mat

    def read_features(self):
        return read_features('Elemental_features.txt')

    def read_vecs(self, atomvec_file):
        f = open(atomvec_file, 'r').readlines()
        fa = open("Atom2Vec/atoms_AE_index.txt", 'r').readlines()
        vectors = [[float(i) for i in i.split()] for i in f]
        atind = [int(i.strip()) for i in fa]
        return vectors, atind
        
    def generateVec_AE(self, envs_mat, k):
        """
        using AE to obtain atoms' features
        """
        self.atom_vecs, _, __  = train_AE(envs_mat, k) 
        print("Atomic encoding is completed.")
    
    def saveAll(self):
        self.saveVec()
        self.saveIndex()
        
    def saveVec(self, filename="magpie_atoms_vec.txt"):
        np.savetxt(filename, self.atoms_vec)
    
    def saveIndex(self, filename="atoms_AE_index.txt"):
        np.savetxt(filename, self.atoms_index, fmt="%d")
    
    
if __name__ == "__main__":
    vec_length = 20
    data_file = "string.json"
    
    atoms_vec = Atom2Vec(data_file, vec_length, 'magpie')
    atoms_vec.saveAll()
