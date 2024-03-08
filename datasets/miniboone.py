from collections import Counter

import numpy as np

import datasets


class MINIBOONE:
    class Data:
        def __init__(self, data):
            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self):
        file = datasets.root + "miniboone/data.npy"
        trn, val, tst = load_data_normalised(file)

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        self.n_dims = self.trn.x.shape[1]


def load_data(root_path):
    # NOTE: To remember how the pre-processing was done.
    # data = pd.read_csv(root_path, names=[str(x) for x in range(50)], delim_whitespace=True)
    # print data.head()
    # data = data.as_matrix()
    # # Remove some random outliers
    # indices = (data[:, 0] < -100)
    # data = data[~indices]
    #
    # i = 0
    # # Remove any features that have too many re-occuring real values.
    # features_to_remove = []
    # for feature in data.T:
    #     c = Counter(feature)
    #     max_count = np.array([v for k, v in sorted(c.iteritems())])[0]
    #     if max_count > 5:
    #         features_to_remove.append(i)
    #     i += 1
    # data = data[:, np.array([i for i in range(data.shape[1]) if i not in features_to_remove])]
    # np.save("~/data/miniboone/data.npy", data)

    data = np.load(root_path)

    N_test = int(0.1 * data.shape[0])
    data_test = data[-N_test:]
    data = data[0:-N_test]
    N_validate = int(0.1 * data.shape[0])
    data_validate = data[-N_validate:]
    data_train = data[0:-N_validate]

    return data_train, data_validate, data_test


def load_data_normalised(root_path):
    data_train, data_validate, data_test = load_data(root_path)
    data = np.vstack((data_train, data_validate))
    mu = data.mean(axis=0)
    s = data.std(axis=0)
    data_train = (data_train - mu) / s
    data_validate = (data_validate - mu) / s
    data_test = (data_test - mu) / s

    return data_train, data_validate, data_test


def load_miniboone(path):
    import pandas as pd

    data = pd.read_csv(
        path, names=[str(x) for x in range(50)], delim_whitespace=True, header=0
    )
    data = data.values

    indices = data[:, 0] < -100
    data = data[~indices]

    i = 0
    # Remove any features that have too many re-occuring real values.
    features_to_remove = []
    for feature in data.T:
        c = Counter(feature)
        max_count = np.array([v for k, v in sorted(c.items())])[0]
        if max_count > 5:
            features_to_remove.append(i)
        i += 1

    data = data[
        :, np.array([i for i in range(data.shape[1]) if i not in features_to_remove])
    ]

    N_test = int(0.1 * data.shape[0])
    data_test = data[-N_test:]
    data = data[0:-N_test]
    N_validate = int(0.1 * data.shape[0])
    data_validate = data[-N_validate:]
    data_train = data[0:-N_validate]

    return data_train, data_validate, data_test


def load_miniboone_normalised(path):
    data_train, data_validate, data_test = load_miniboone(path)
    data = np.vstack((data_train, data_validate))
    mu = data.mean(axis=0)
    s = data.std(axis=0)
    data_train = (data_train - mu) / s
    data_validate = (data_validate - mu) / s
    data_test = (data_test - mu) / s

    return data_train, data_validate, data_test


if __name__ == "__main__":
    import torch
    from lakde import *

    data = MINIBOONE()
    X_train = torch.from_numpy(data.trn.x).cuda()
    Y_test = torch.from_numpy(data.tst.x).cuda()
    Y_val = torch.from_numpy(data.val.x).cuda()

    # X_train, Y_test, Y_val = load_miniboone_normalised(datasets.root + 'miniboone/MiniBooNE_PID.txt')
    # X_train = torch.from_numpy(X_train).float().cuda()
    # Y_test = torch.from_numpy(Y_test).float().cuda()
    # Y_val = torch.from_numpy(Y_val).float().cuda()

    print(X_train.shape)
    print(Y_test.shape)
    print(Y_val.shape)

    k = 250
    nu_0 = 80

    print("k =", k)
    print("nu_0 =", nu_0)

    # subset_size = 10_000

    mean_lls = []

    for i in range(5):
        # model = LocalFullKDE(nu_0=k, k_or_knn_graph=k, global_sigma_0=False, block_size=4000, verbose=True)

        # model = LocalFullGlobalKDE(nu_0=nu_0, k_or_knn_graph=k, global_sigma_0=False, block_size=2000, verbose=True)
        # model = HierarchicalKDE(nu_0=44, k_or_knn_graph=40, block_size=2000, verbose=True)
        # model = SharedFullKDE(verbose=True)
        # model = SharedDiagonalizedKDE(verbose=True)
        model = SharedScalarKDE(verbose=True)

        # subsample X_train
        # np.random.seed(i)
        # inds = np.random.choice(len(X_train), subset_size, replace=False)
        # X = X_train[inds]
        X = X_train

        print("Training...")
        model.fit(X, iterations=20, Y=Y_val, validate_every=1)  # use_sparse_ops=True

        lls = model.log_pred_density(X, Y_val)
        mean_lls.append(lls.mean())
        print(mean_lls[-1])

    mean_lls = torch.stack(mean_lls)
    print(torch.mean(mean_lls))
    print(torch.std(mean_lls))
