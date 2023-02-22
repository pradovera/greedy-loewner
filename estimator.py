import numpy as np

class estimator:
    kind = "NONE"
    def __init__(self, tol:float, engine):
        self.tol = tol
        self.engine = engine

    def indicator(self, z_test, approx, *args, **kwargs):
        return 1 / np.abs(approx(z_test, only_den = True))

    def setup(self, *args, **kwargs): pass
    def pre_check(self, *args, **kwargs): pass
    def mid_setup(self, *args, **kwargs): pass
    def post_check(self, *args, **kwargs): pass
    def build_eta(self, *args, **kwargs): pass

class estimatorLookAhead(estimator):
    kind = "LOOK_AHEAD"

    def mid_setup(self, z_test, idx_next, *args, **kwargs):
        self.z = z_test[idx_next]

    def post_check(self, sample, approx, *args, **kwargs):
        self.error = (np.linalg.norm(approx(self.z) - sample, axis = 1)
                    / np.linalg.norm(sample, axis = 1))
        return 1 * (self.error < self.tol)

    def build_eta(self, z_test, approx, *args, **kwargs):
        indicator = self.indicator(z_test, approx)
        idx = np.argmax(indicator)
        self.mid_setup(z_test, idx)
        sample = self.engine.sample(1j * self.z, verbose = 0)
        self.post_check(sample, approx)
        return self.error * indicator / indicator[idx]

class estimatorLookAheadBatch(estimator):
    kind = "LOOK_AHEAD_BATCH"

    def __init__(self, tol:float, engine, N:int):
        super().__init__(tol, engine)
        self.N = N

    def mid_setup(self, z_test, idx_next, indicator, approx, *args, **kwargs):
        ind = np.array(indicator)
        idx = [idx_next]
        for n in range(self.N - 1):
            ind *= np.abs(z_test - z_test[idx[-1]])
            idx += [np.argmax(ind)]
        self.z = np.array([z_test[j] for j in idx])

    def post_check(self, sample, approx, *args, **kwargs):
        samples = np.vstack((sample, self.engine.sample(1j * self.z[1 :],
                                                        verbose = 0)))
        error = (np.linalg.norm(approx(self.z) - samples, axis = 1)
               / np.linalg.norm(samples, axis = 1))
        idx = np.argmax(error)
        self.error_z = self.z[idx]
        self.error = error[idx]
        return 1 * (self.error < self.tol)

    def build_eta(self, z_test, approx, *args, **kwargs):
        indicator = self.indicator(z_test, approx)
        self.mid_setup(z_test, np.argmax(indicator), indicator, approx)
        sample = self.engine.sample(1j * self.z[0], verbose = 0)
        self.post_check(sample, approx)
        return self.error * indicator / self.indicator(self.error_z, approx)

class estimatorRandom(estimator):
    kind = "RANDOM"

    def __init__(self, tol:float, engine, N:int, seed:int):
        super().__init__(tol, engine)
        self.N = N
        self.seed = seed

    def setup(self, z_range, *args, **kwargs):
        np.random.seed(self.seed)
        self.z = 10 ** (np.log10(z_range[0])
                      + (np.log10(z_range[1]) - np.log10(z_range[0]))
                      * np.random.rand(self.N))
        self.samples = self.engine.sample(1j * self.z, verbose = 0)
        self.samples_norm = np.linalg.norm(self.samples, axis = 1)

    def pre_check(self, approx, *args, **kwargs):
        error = (np.linalg.norm(approx(self.z) - self.samples,
                                axis = 1) / self.samples_norm)
        idx = np.argmax(error)
        self.error_z = self.z[idx]
        self.error = error[idx]
        return 1 * (self.error < self.tol)

    def build_eta(self, z_test, approx, *args, **kwargs):
        indicator = self.indicator(z_test, approx)
        self.pre_check(approx)
        return self.error * indicator / self.indicator(self.error_z, approx)
