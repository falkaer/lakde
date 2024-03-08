import signal

import torch
from torch.utils.tensorboard.summary import hparams
from torch.utils.tensorboard.writer import SummaryWriter


def tensor_as(x, other):
    if torch.is_tensor(x):
        return x.to(dtype=other.dtype, device=other.device)
    return torch.tensor(x, dtype=other.dtype, device=other.device)


def sample_multinomial(total_count, probs):
    device = probs.device
    probs = probs.cpu()
    total_count = probs.new_full((), total_count)
    remainder = probs.new_ones(())
    sample = torch.empty_like(probs, dtype=torch.long)
    for i, p in enumerate(probs):
        s = torch.binomial(total_count, p / remainder)
        sample[i] = s
        total_count -= s
        remainder -= p
    return sample.to(device)


def cov(X, dtype=None, inplace=False):
    N = X.size(-2)
    if dtype is None:
        dtype = torch.promote_types(X.dtype, torch.float32)
    X_bar = X.mean(dim=-2, dtype=dtype, keepdim=True)
    if inplace:
        X_center = X.sub_(X_bar)
    else:
        X_center = X.sub(X_bar)
    return torch.matmul(X_center.mT, X_center).div_(N - 1)


def batched_knn_cov(X, X_self, knn_graph, include_self=True, self_center=True):
    D = X.size(1)
    M, K = knn_graph.shape
    out = X.new_empty(M, D, D)

    # the output scales with N * D * D, but we don't want intermediate results
    # to scale with K, so redistribute the batch size
    bsize = max(M * D // K, 128)

    for i in range(0, M, bsize):
        if include_self and not self_center:
            graph = knn_graph[i : i + bsize]
            Xi = X.new_empty(graph.size(0), K + 1, D)
            Xi[:, 0] = X_self[i : i + bsize]
            Xi[:, 1 : K + 1] = X[graph]
        else:
            Xi = X[knn_graph[i : i + bsize]]

        if self_center:
            X_bar = X_self[i : i + bsize, None]
        else:
            X_bar = Xi.mean(dim=1, keepdim=True)
        X_center = Xi.sub_(X_bar)
        torch.bmm(X_center.mT, X_center, out=out[i : i + bsize])

    return out.div_(K if self_center or include_self else K - 1)


def var(X):
    N = X.size(-2)
    # mintype float32
    dtype = torch.promote_types(X.dtype, torch.float)
    X_bar = X.mean(dim=-2, dtype=dtype, keepdim=True)
    X_center = X - X_bar
    return torch.sum(X_center**2, dim=-2) / (N - 1)


def _compute_knn_graph_generic(
    X,
    k,
    bsize=512,
    return_dists=False,
    include_self=False,
    metric="l2",
    dtype=torch.int32,
):
    N = X.size(0)
    metric = metric.lower()
    I = X.new_empty(N, k, dtype=dtype)

    if metric == "l2":
        p = 2
    elif metric == "l1":
        p = 1
    else:
        raise ValueError("unknown metric")

    if return_dists:
        D = X.new_empty(N, k)

    for i in range(0, N, bsize):
        Xt = X[i : i + bsize]
        dists = torch.cdist(Xt, X, p=p)

        if not include_self:
            torch.diagonal(dists, i).fill_(float("inf"))

        if return_dists:
            D[i : i + bsize], I[i : i + bsize] = dists.topk(
                k, dim=1, largest=False, sorted=True
            )
        else:
            I[i : i + bsize] = dists.topk(k, dim=1, largest=False, sorted=True)[1]

    if return_dists:
        return D, I
    return I


try:
    import faiss

    def swig_ptr_from_FloatTensor(x):
        assert x.is_contiguous()
        assert x.dtype == torch.float32
        return faiss.cast_integer_to_float_ptr(
            x.storage().data_ptr() + x.storage_offset() * 4
        )

    def swig_ptr_from_Int64Tensor(x):
        assert x.is_contiguous()
        assert x.dtype == torch.int64
        return faiss.cast_integer_to_idx_t_ptr(
            x.storage().data_ptr() + x.storage_offset() * 8
        )

    def swig_ptr_from_Int32Tensor(x):
        assert x.is_contiguous()
        assert x.dtype == torch.int32
        return faiss.cast_integer_to_idx_t_ptr(
            x.storage().data_ptr() + x.storage_offset() * 4
        )

    def swig_ptr_from_IntTensor(x):
        assert x.dtype in [torch.int32, torch.int64]
        if x.dtype == torch.int32:
            return swig_ptr_from_Int32Tensor(x)
        else:
            return swig_ptr_from_Int64Tensor(x)

    def compute_knn_graph(
        X,
        k,
        bsize=512,
        return_dists=False,
        include_self=False,
        metric="l2",
        dtype=torch.int32,
    ):
        N = X.size(0)
        metric = metric.lower()
        temp_memory = 512 * 1024 * 1024
        res = faiss.StandardGpuResources()
        res.setTempMemory(temp_memory)
        res.setDefaultNullStreamAllDevices()

        k = k if include_self else k + 1
        D = X.new_empty(N, k, dtype=torch.float32)
        I = X.new_empty(N, k, dtype=dtype)

        if X.is_contiguous():
            row_major = True
        elif X.T.is_contiguous():
            X = X.T
            row_major = False
        else:
            X = X.contiguous()
            row_major = True

        X_ptr = swig_ptr_from_FloatTensor(X)
        dists_ptr = swig_ptr_from_FloatTensor(D)
        I_ptr = swig_ptr_from_IntTensor(I)

        args = faiss.GpuDistanceParams()
        if metric == "l2":
            args.metric = faiss.METRIC_L2
        elif metric == "l1":
            args.metric = faiss.METRIC_L1
        else:
            raise ValueError("unknown metric")

        args.k = k
        args.dims = X.size(1)
        args.vectors = X_ptr
        args.vectorsRowMajor = row_major
        args.numVectors = N
        args.queries = X_ptr
        args.queriesRowMajor = row_major
        args.numQueries = N
        args.outDistances = dists_ptr
        args.outIndices = I_ptr
        if dtype == torch.int32:
            args.outIndicesType = faiss.IndicesDataType_I32
        elif dtype == torch.int64:
            args.outIndicesType = faiss.IndicesDataType_I64

        faiss.bfKnn(res, args)

        if not include_self:
            D = D[:, 1:]
            I = I[:, 1:]

        if return_dists:
            return D, I

        return I

except ImportError:
    compute_knn_graph = _compute_knn_graph_generic


class SaneSummaryWriter(SummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict, hparam_domain_discrete=None):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict:
            raise TypeError("hparam_dict should be dictionary.")

        # since we are adding the summaries to the current writer
        # only the keys of the metric_dict will be used, just pass empty values
        exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)

        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)

        for k, v in metric_dict.items():
            if v is not None:
                self.add_scalar(k, v)


class InterruptDelay:
    def __init__(self):
        self.signal_received = None
        self.old_handler = None

    def __enter__(self):
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        print("Interrupt received during update, finishing up...")
        self.signal_received = (sig, frame)

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received is not None:
            self.old_handler(*self.signal_received)
