# Think Global, Adapt Local: Learning Locally Adaptive Kernel Density Estimators


This repository contains source code for training and evaluating variants of locally adaptive KDEs wherein every kernel is parameterized by its own full covariance matrix in contrast to traditional KDEs where kernel structure is shared across the entire input space. This allows for a flexible characterization of the data density.

Training is made tractable using variational inference to derive a sparse latent variable representation of the parameters and using custom CUDA kernels for large memory and computation savings. This repository includes implementations of two such models (`LocalKNearestKDE` and `LocalHierarchicalKDE`) which are locally adaptive, and three conventional shared-kernel KDE models for comparisons (`SharedFullKDE`, `SharedDiagonalizedKDE` and `SharedScalarKDE`):

* `LocalKNearestKDE`: A locally adaptive KDE where every input point learns its own posterior covariance. Prior covariance is based on the covariance structure of the K nearest neighbors of each point.
* `LocalHierarchicalKDE`: A locally adaptive KDE where every input point learns its own posterior covariance. Prior covariance is a learned shared covariance factor. 
* `SharedFullKDE`: A shared-kernel KDE where every point shares a posterior covariance. Prior covariance is set to the empirical covariance.
* `SharedDiagonalizedKDE`: A restricted version of `SharedFullKDE` where the covariance is a diagonal matrix.
* `SharedScalarKde`: A further restricted version of `SharedFullKDE` where the covariance is a diagonal matrix with the same scalar along the diagonal.

The models have a sklearn-like API with a `fit(self, data, iterations)` function that initializes the model and takes `iterations` optimization steps. After fitting a model, `log_pred_density(self, data, new_data)` can be used to get the predictive log-likelihood (log density) of `new_data`. See also `experiments/run_kde`.

## Prerequisites

1. PyTorch >= 1.8.1
2. NumPy
3. h5py and Pandas to use tabular data sets
4. sklearn to use toy data sets
5. ax for hyperparameter search (optional)
6. faiss can be used for faster KNN computation (optional)

To use custom CUDA kernels you will need nvcc installed and a not-too-recent C++ compiler (tested with GCC 7). The C++ compiler can be selected by setting the CXX environment variable and the CUDA installation by setting CUDA_HOME. Use of custom CUDA kernels can be disabled entirely by setting LAKDE_NO_EXT=1.

## Preparing datasets

The tabular datasets are prepared as described [here](https://github.com/gpapamak/maf) and placed in `data/`, e.g. BSDS300 data should be unpacked to `data/bsds300`.
