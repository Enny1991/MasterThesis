import numpy as np

# Simple routine to do denoising PCA on a 2-D vector
# A given 2-D matrix in project on a low dimensional space corresponding to the given variance explained
# and then reprojected back in the original space.
#
# Use:
# Given the matrix data of shape (n_samples x n_dimensions)
# data_hat = denoise(data, var_exp=0.9)


def project(high_dim, v, dims):
    v_low = v[:, :dims]
    return np.dot(high_dim, v_low).T


def reproject(low_dim, v, pad):
    print low_dim.shape
    print v.shape
    filler = np.zeros((low_dim.shape[0], pad))
    to_mult = np.concatenate((low_dim, filler), axis=1)
    print to_mult.shape
    return np.dot(to_mult, v.T).T


def denoise(x, var_exp=0.9):
    dims = x.shape[1]  # dimension of the original space

    U, S, V = np.linalg.svd(x)  # SVD to understand the variance explained by every dimension

    # Calculating how many dimensions are needed to for the given variance explained
    sv_sum = np.sum(S)
    red_dim = None
    for i, k in enumerate(np.cumsum(S) / sv_sum):
        if k >= var_exp:
            red_dim = i + 1
            break

    low_dim = project(x, V, red_dim)  # projection in the lower dimensional space
    high_dim_denoise = reproject(low_dim.T, V, dims - red_dim)  # Reprojection in the original space

    return high_dim_denoise.T, red_dim

