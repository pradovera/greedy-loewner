import numpy as np
import numpy.linalg as npla
import scipy.sparse.linalg as scspla
import scipy.sparse as sp
from scipy.io import loadmat
from estimator import estimatorLookAhead, estimatorLookAheadBatch, estimatorRandom

def load_example(name):
    if name in ["MNA_4", "MNA_4_RANDOM"]:
        engine = slicotEngine("MNA_4.mat")
    if name in ["TLINE", "TLINE_MEMORY"]:
        engine = slicotEngine("tline.mat")
    if name in ["ISS", "ISS_BATCH"]:
        engine = slicotEngine("iss.mat")
    if name in ["MNA_4", "TLINE", "TLINE_MEMORY", "ISS"]:
        estimator = estimatorLookAhead(1e-3, engine)
    if name in ["ISS_BATCH"]:
        estimator = estimatorLookAheadBatch(1e-3, engine, 5)
    if name in ["MNA_4_RANDOM"]:
        estimator = estimatorRandom(1e-3, engine, 100, 42)
    N_test = 10000
    if name in ["MNA_4", "MNA_4_RANDOM", "TLINE"]:
        N_memory = 1
    if name in ["TLINE_MEMORY", "ISS", "ISS_BATCH"]:
        N_memory = 3
    return engine, 100, estimator, N_test, N_memory

z_ranges = {"MNA_4.mat":[3e4, 3e9],
            "tline.mat":[1e7, 1e15],
            "iss.mat":[1e-2, 1e3]}

class slicotEngine:
    def __init__(self, name:str):
        # load SLICOT matrices from file
        self.npar, self.nAs, self.nbs = 1, 2, 1
        data = loadmat("./SLICOT data/" + name)
        if isinstance(data["A"], np.ndarray): # dense matrices
            self._solver = lambda A, b: npla.solve(A, b)
            if "E" not in data.keys():
                data["E"] = np.eye(data["A"].shape[0])
        else: # sparse matrices
            self._solver = lambda A, b: scspla.spsolve(A, b,
                                                       use_umfpack = False)
            if "E" not in data.keys():
                data["E"] = sp.eye(data["A"].shape[0],
                                   format = data["A"].format)
            if "B" in data.keys() and not isinstance(data["B"], np.ndarray):
                data["B"] = data["B"].toarray()
            if "C" in data.keys() and not isinstance(data["C"], np.ndarray):
                data["C"] = data["C"].todense()
        if "B" not in data.keys():
            data["B"] = np.ones((data["A"].shape[0], 1))
        if "C" not in data.keys():
            data["C"] = data["B"].T.conj()
        if "D" not in data.keys():
            data["D"] = 0.
        self.A = data["A"] + 0.j
        self.E = data["E"] + 0.j
        self.B = np.asarray(data["B"] + 0.j)
        self.C = np.asarray(data["C"] + 0.j)
        self.D = data["D"]
        self.z_range = z_ranges[name] # frequency range

    @property
    def size(self):
        return self.C.shape[0] * self.B.shape[1]

    def sample(self, z:float, verbose : bool = 1):
        while hasattr(z, "__len__"): # evaluation at multiple frequencies
            if len(z) > 1:
                out = np.empty((len(z), self.size), dtype = complex)
                for j, z_j in enumerate(z):
                    out[j] = self.sample(z_j, verbose)
                return out
            z = z[0]
        if verbose: print("sampling at z={}".format(z))
        x = self._solver(z * self.E - self.A, self.B) # get state
        return self.C.dot(x).reshape(1, -1) + self.D # get output
