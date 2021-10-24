import torch

__all__ = [
    "SamplePredictor",
]


class SamplePredictor:
    """Multi sample prediction using a trained `NeuralProcess` model. `is_dist` if should return the 
    predictive instead of loc (e.g. mean)."""

    def __init__(self, model, is_dist=False,is_enc=False):
        self.model = model
        self.is_dist = is_dist
        self.is_enc = is_enc
    def __call__(self, *args):
        # q_zCc.base_dist.scale.mean()
        if self.is_enc:
            return self.model(*args)

        p_y_pred, *_ = self.model(*args)

        if self.is_dist:
            return p_y_pred

        loc_ys = p_y_pred.base_dist.loc.detach()
        return loc_ys
