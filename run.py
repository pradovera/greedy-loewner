import sys
import numpy as np
from matplotlib import pyplot as plt
from load import load_example
from barycentric import barycentricFunction

#%% choose example
allowed_tags = ["MNA_4", "MNA_4_RANDOM",
                "TLINE", "TLINE_MEMORY",
                "ISS", "ISS_BATCH", "ISS_RANDOM"]
if len(sys.argv) > 1:
    example_tag = sys.argv[1]
else:
    example_tag = input(("Input example_tag:\n(Allowed values: {})\n").format(
                                                                 allowed_tags))
example_tag = example_tag.upper().replace(" ","").strip()
if example_tag not in allowed_tags:
    raise Exception(("Value of example_tag not recognized. Allowed values:\n"
                     "{}").format(allowed_tags))

#%% load example
engine, Smax, estimator, N_test, N_memory = load_example(example_tag)
z_range, size = engine.z_range, engine.size
sizeh = int(np.round(size ** .5)) # the transfer function is square (p = m)
# initialize test set
z_test = list(np.geomspace(*z_range, N_test))
# estimator setup (only for RANDOM)
estimator.setup(z_range)

#%% get initial sample
z_sample = z_test.pop(0) # remove initial sample point from test set
sample = engine.sample(1j * z_sample)
print("0: sampled at z={}".format(z_sample))
approx = barycentricFunction(np.array([z_sample]), np.ones(1), sample)
L = .5j * (sample.conj() - sample).T / z_sample # initial Loewner matrix

#%% adaptivity loop
n_memory = 0
for _ in range(Smax): # max number of samples
    # estimator pre-check (only for RANDOM)
    flag = estimator.pre_check(approx)
    if flag == 0: n_memory = 0 # error is too large
    if flag == 1: n_memory += 1 # error is below tolerance

    # termination check
    if n_memory >= N_memory: break # enough small errors in a row

    # find next sample point
    indicator = estimator.indicator(z_test, approx)
    idx_sample = np.argmax(indicator)

    # estimator mid-setup (only for BATCH)
    estimator.mid_setup(z_test, idx_sample, indicator, approx)

    z_sample = z_test.pop(idx_sample) # remove sample point from test set
    sample = engine.sample(1j * z_sample) # compute new sample
    
    # estimator post-check (only for LOOK_AHEAD and BATCH)
    flag = estimator.post_check(sample, approx)
    if flag == 0: n_memory = 0 # error is too large
    if flag == 1: n_memory += 1 # error is below tolerance
    print("{}: sampled at z={}".format(approx.nsupp, z_sample))
    
    # update surrogate with new support points and values
    approx.supp = np.append(approx.supp, z_sample)
    approx.vals = np.append(approx.vals, sample, axis = 0)

    # update Loewner matrix
    L = np.pad(L, [(0, size), (0, 1)])
    for j, (k, s) in enumerate(zip(approx.supp, approx.vals)):
        if j < approx.nsupp - 1: # right column
            L[j * size : (j + 1) * size, [-1]] = (1j * (s.conj() - sample).T
                                                     / (k + z_sample))
        else: # bottom row
            L[j * size : (j + 1) * size, :] = (1j * (s.conj() - approx.vals).T
                                                  / (z_sample + approx.supp))

    # update surrogate with new barycentric coefficients
    approx.coeffs = np.linalg.svd(np.linalg.qr(L)[1])[2][-1].conj()

print("greedy loop terminated at {} samples".format(approx.nsupp))

#%% predict and compute errors
z_post = np.geomspace(*z_range, 101)
H_exact = engine.sample(1j * z_post)
H_approx = approx(z_post)
H_err = estimator.compute_error(H_approx, H_exact) + 1e-16
eta = estimator.build_eta(z_test, approx)

#%% plots
plt.figure()
plt.loglog(z_post, np.abs(H_exact))
plt.loglog(z_post, np.abs(H_approx), '--')
plt.loglog(approx.supp, np.abs(approx.vals), 'o')
plt.legend(["H{}{}".format(i + 1, j + 1) for i in range(sizeh)
                                         for j in range(sizeh)])
plt.xlabel("Im(z)"), plt.xlabel("|H|")
plt.figure()
plt.loglog(z_post, H_err)
plt.loglog(z_test, eta, ":")
plt.loglog(z_range, [estimator.tol] * 2, '--')
plt.legend(["error", "estimator"])
plt.xlabel("Im(z)"), plt.xlabel("relative error")
plt.show()
