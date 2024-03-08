from .local_knn_kde import LocalKNearestKDE


# just ignore the nu_0 and set it very high
class LocalKNearestKDEInfNu(LocalKNearestKDE):
    def __init__(self, nu_0, *args, **kwargs):
        super().__init__(100_000, *args, **kwargs)

    def hparam_state_dict(self):
        d = super().hparam_state_dict()
        d["model"] = "knn_inf_nu"
        return d
