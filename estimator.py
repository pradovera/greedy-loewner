import numpy as np

class estimator:
    # class that implements error indicator and estimator
    def __init__(self, tol:float, delta:float, engine):
        self.tol = tol       # greedy tolerance
        self.delta = delta   # cutoff in definition of relative error
        self.engine = engine # high-fidelity engine for sampling

    def compute_error(self, app, ex, ex_norm = None):
        # compute adjusted relative error
        if ex_norm is None:
            ex_norm = np.linalg.norm(ex, axis = 1)
        return np.linalg.norm(app - ex, axis = 1) / (ex_norm + self.delta)

    def indicator(self, z_test, approx, *args, **kwargs):
        # reciprocal of magnitude of surrogate denominator
        return 1 / np.abs(approx(z_test, only_den = True))

    def setup(self, *args, **kwargs): pass
    def pre_check(self, *args, **kwargs): pass
    def mid_setup(self, *args, **kwargs): pass
    def post_check(self, *args, **kwargs): pass
    def build_eta(self, *args, **kwargs): pass

class estimatorLookAhead(estimator):
    def mid_setup(self, z_test, idx_next, *args, **kwargs):
        # next sample point
        self.z = z_test[idx_next]

    def post_check(self, sample, approx, *args, **kwargs):
        # error at next sample point
        self.error = self.compute_error(approx(self.z), sample)
        return 1 * (self.error < self.tol)

    def build_eta(self, z_test, approx, *args, **kwargs):
        # evaluate error estimator
        indicator = self.indicator(z_test, approx)
        idx = np.argmax(indicator)
        self.mid_setup(z_test, idx)
        sample = self.engine.sample(1j * self.z)
        self.post_check(sample, approx)
        return self.error * indicator / indicator[idx]

class estimatorLookAheadBatch(estimator):
    def __init__(self, tol:float, delta:float, engine, N:int):
        super().__init__(tol, delta, engine)
        self.N = N # batch size

    def mid_setup(self, z_test, idx_next, indicator, approx, *args, **kwargs):
        # next sample point and test points
        ind = np.array(indicator)
        self.z_idx = [idx_next]
        for n in range(self.N - 1):
            ind *= np.abs(z_test - z_test[self.z_idx[-1]])
            self.z_idx += [np.argmax(ind)]
        self.z = np.array([z_test[j] for j in self.z_idx])

    def post_check(self, sample, approx, *args, **kwargs):
        # error at next sample point and at test points
        print("computed {} extra test samples".format(self.N - 1))
        samples = np.vstack((sample, self.engine.sample(1j * self.z[1 :])))
        error = self.compute_error(approx(self.z), samples)
        idx = np.argmax(error)
        self.error_z = self.z[idx]
        self.error = error[idx]
        return 1 * (self.error < self.tol)

    def build_eta(self, z_test, approx, *args, **kwargs):
        # evaluate error estimator
        indicator = self.indicator(z_test, approx)
        self.mid_setup(z_test, np.argmax(indicator), indicator, approx)
        sample = self.engine.sample(1j * self.z[0])
        self.post_check(sample, approx)
        return self.error * indicator / self.indicator(self.error_z, approx)

class estimatorRandom(estimator):
    def __init__(self, tol:float, delta:float, engine, N:int, seed:int):
        super().__init__(tol, delta, engine)
        self.N = N       # sample size
        self.seed = seed # random seed

    def setup(self, z_range, *args, **kwargs):
        # compute test points and test samples
        np.random.seed(self.seed)
        self.z = 10 ** (np.log10(z_range[0])
                      + (np.log10(z_range[1]) - np.log10(z_range[0]))
                      * np.random.rand(self.N))
        self.samples = self.engine.sample(1j * self.z)
        print("computed {} extra test samples".format(self.N))
        self.samples_norm = np.linalg.norm(self.samples, axis = 1)

    def pre_check(self, approx, *args, **kwargs):
        # error at test points
        error = self.compute_error(approx(self.z), self.samples,
                                   self.samples_norm)
        idx = np.argmax(error)
        self.error_z = self.z[idx]
        self.error = error[idx]
        return 1 * (self.error < self.tol)

    def build_eta(self, z_test, approx, *args, **kwargs):
        # evaluate error estimator
        indicator = self.indicator(z_test, approx)
        self.pre_check(approx)
        return self.error * indicator / self.indicator(self.error_z, approx)
