from . import LocalKNearestKDEFixedTau


class LocalKNearestKDEPriorOnly(LocalKNearestKDEFixedTau):
    def __init__(self, nu_0, *args, **kwargs):
        super().__init__(100_000, *args, **kwargs)

    def hparam_state_dict(self):
        d = super().hparam_state_dict()
        d["model"] = "knn_fixed_tau_inf_nu"
        return d
