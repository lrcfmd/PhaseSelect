from scipy.sparse.linalg import svds
from EnvMatrix import EnvsMat
import numpy as np
from AE import train_AE

class Atom2Vec:
    def __init__(self, filename, k):
        envs_mat = EnvsMat(filename)
        self.atoms_index = envs_mat.atoms
        envs_mat = envs_mat.envs_mat
        print(envs_mat.shape)
        print(envs_mat[0])
        sys.exit()
 
        self.atoms_vec = self.generateVec_AE(envs_mat, k)
        #self.atoms_vec = self.generateVec(envs_mat, k)
        
    def generateVec(self, envs_mat, k):
        """
        using svd to obtain atoms' features
        """
        print("SVD -- ", end="")
        u, d, v = svds(envs_mat, k=k, which="LM")
        print("Complete!")
        return u @ np.diag(d)

    def generateVec_AE(self, envs_mat, k):
        """
        using AE to obtain atoms' features
        """
        print ("AE --", end="")
        vec, re = train_AE(envs_mat, k)
        print("Complete!")
        return vec
    
    def saveAll(self):
        self.saveVec()
        self.saveIndex()
        
    def saveVec(self, filename="atoms_AE_vec.txt"):
        np.savetxt(filename, self.atoms_vec)
    
    def saveIndex(self, filename="atoms_AE_index.txt"):
        np.savetxt(filename, self.atoms_index, fmt="%d")
    
    
if __name__ == "__main__":
    vec_length = 20
    data_file = "string.json"
    
    atoms_vec = Atom2Vec(data_file, vec_length)
    atoms_vec.saveAll()
