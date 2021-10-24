"""Module for all the loss of Neural Process Family."""
import abc
import math
import numpy as np
import torch
import torch.nn as nn
from npf.utils.helpers import (
    LightTailPareto,
    dist_to_device,
    logcumsumexp,
    sum_from_nth_dim,
)
from torch.distributions.kl import kl_divergence

__all__ = ["CNPFLoss","Contrastive_CNPFLoss","FCLR_Loss", "ELBOLossLNPF", "SUMOLossLNPF", "NLLLossLNPF"]


def sum_log_prob(prob, sample):
    """Compute log probability then sum all but the z_samples and batch."""
    # size = [n_z_samples, batch_size, *]
    #print(sample.shape)
    log_p = prob.log_prob(sample)
    # size = [n_z_samples, batch_size]
    sum_log_p = sum_from_nth_dim(log_p, 2)
    return sum_log_p


class BaseLossNPF(nn.Module, abc.ABC):
    """
    Compute the negative log likelihood loss for members of the conditional neural process (sub-)family.

    Parameters
    ----------
    reduction : {None,"mean","sum"}, optional
        Batch wise reduction.

    is_force_mle_eval : bool, optional
        Whether to force mac likelihood eval even if has access to q_zCct
    """

    def __init__(self, reduction="mean", is_force_mle_eval=True,is_contrastive=False,device='cpu',batch_size=1,lreg=None):
        super().__init__()
        self.reduction = reduction
        self.is_contrastive = is_contrastive
        self.is_force_mle_eval = is_force_mle_eval
        self.lreg = lreg
        
        if self.is_contrastive:
            self.device = device
            self.batch_size = batch_size
            self.nt_xent_criterion = NTXentLoss(self.device, self.batch_size,temperature = 0.5,use_cosine_similarity=True)

    def forward(self, pred_outputs, Y_trgt,is_contrastive=False):
        """Compute the Neural Process Loss.

        Parameters
        ----------
        pred_outputs : tuple
            Output of `NeuralProcessFamily`.

        Y_trgt : torch.Tensor, size=[batch_size, *n_trgt, y_dim]
            Set of all target values {y_t}.

        Return
        ------
        loss : torch.Tensor
            size=[batch_size] if `reduction=None` else [1].
        """
        if self.is_contrastive:
            p_yCc, z_samples, q_zCc, q_zCct,R = pred_outputs
        else:
            p_yCc, z_samples, q_zCc, q_zCct = pred_outputs

        if self.is_contrastive:
            loss = self.get_loss(p_yCc, z_samples, q_zCc, q_zCct, Y_trgt,R,self.nt_xent_criterion)
        else:
            loss = self.get_loss(p_yCc, z_samples, q_zCc, q_zCct, Y_trgt)

        if self.reduction is None:
            # size = [batch_size]
            return loss
        elif self.reduction == "mean":
            # size = [1]
            return loss.mean(0)
        elif self.reduction == "sum":
            # size = [1]
            return loss.sum(0)
        else:
            raise ValueError(f"Unknown {self.reduction}")

    @abc.abstractmethod
    def get_loss(self, p_yCc, z_samples, q_zCc, q_zCct, Y_trgt):
        """Compute the Neural Process Loss

        Parameters
        ------
        p_yCc: torch.distributions.Distribution, batch shape=[n_z_samples, batch_size, *n_trgt] ; event shape=[y_dim]
            Posterior distribution for target values {p(Y^t|y_c; x_c, x_t)}_t

        z_samples: torch.Tensor, size=[n_z_samples, batch_size, *n_lat, z_dim]
            Sampled latents. `None` if `encoded_path==deterministic`.

        q_zCc: torch.distributions.Distribution, batch shape=[batch_size, *n_lat] ; event shape=[z_dim]
            Latent distribution for the context points. `None` if `encoded_path==deterministic`.

        q_zCct: torch.distributions.Distribution, batch shape=[batch_size, *n_lat] ; event shape=[z_dim]
            Latent distribution for the targets. `None` if `encoded_path==deterministic`
            or not training or not `is_q_zCct`.

        Y_trgt: torch.Tensor, size=[batch_size, *n_trgt, y_dim]
            Set of all target values {y_t}.

        Return
        ------
        loss : torch.Tensor, size=[1].
        """
        pass


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity=True):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        zis = nn.functional.normalize(zis, dim=1)
        zjs = nn.functional.normalize(zjs, dim=1)
        
        representations = torch.cat([zjs, zis], dim=0)
        similarity_matrix = self.similarity_function(representations, representations)
        #print(similarity_matrix)
        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        
        #print(l_pos.shape,r_pos.shape,self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)
        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        
        
        loss = self.criterion(logits, labels)
        return loss / (2 * self.batch_size)

class FCLR_Loss(BaseLossNPF):

    def get_loss(self, p_yCc, _, q_zCc, ___, Y_trgt,R,nt_xent_criterion):
        sum_log_p_yCz = sum_log_prob(p_yCc, Y_trgt)
        nll = -sum_log_p_yCz.squeeze(0)

        if len(R)<self.batch_size*2:
            return nll

        zis = R[:self.batch_size]
        zjs = R[self.batch_size:]
        nt_xent = nt_xent_criterion(zis, zjs)

        return nt_xent

class Contrastive_CNPFLoss(BaseLossNPF):
    """Losss for conditional neural process (suf-)family [1]."""

    def get_loss(self, p_yCc, _, q_zCc, ___, Y_trgt,R,nt_xent_criterion):
        sum_log_p_yCz = sum_log_prob(p_yCc, Y_trgt)
        nll = -sum_log_p_yCz.squeeze(0)

        #print(R.shape)
        r_batch = R.shape[0]//2
        zis = R[:r_batch]
        zjs = R[r_batch:]

        if r_batch < self.batch_size*2:
            diff = self.batch_size - r_batch
            ids = np.random.randint(0,r_batch,diff)
            zis = torch.cat([zis,zis[ids]])
            zjs = torch.cat([zjs,zjs[ids]])
            
        nt_xent = nt_xent_criterion(zis, zjs)
        #print(nll,nt_xent)
        return nll*self.lreg + nt_xent

class CNPFLoss(BaseLossNPF):
    """Losss for conditional neural process (suf-)family [1]."""

    def get_loss(self, p_yCc, _, q_zCc, ___, Y_trgt):
        assert q_zCc is None
        # \sum_t log p(y^t|z)
        # \sum_t log p(y^t|z). size = [z_samples, batch_size]
        #print(Y_trgt.shape,torch.mean(Y_trgt))
        #print(Y_trgt[0].shape,torch.mean(Y_trgt[0]))
        #print(Y_trgt[0,:,0].shape,torch.mean(Y_trgt[0,:,0]))
        #print('')
        
        sum_log_p_yCz = sum_log_prob(p_yCc, Y_trgt)
        #print(sum_log_p_yCz)
        # size = [batch_size]
        nll = -sum_log_p_yCz.squeeze(0)
        #print(nll)
        return nll


class ELBOLossLNPF(BaseLossNPF):
    """Approximate conditional ELBO [1].

    References
    ----------
    [1] Garnelo, Marta, et al. "Neural processes." arXiv preprint
        arXiv:1807.01622 (2018).
    """

    def get_loss(self, p_yCc, _, q_zCc, q_zCct, Y_trgt,R=None,nt_xent_criterion=None):

        # first term in loss is E_{q(z|y_cntxt,y_trgt)}[\sum_t log p(y^t|z)]
        # \sum_t log p(y^t|z). size = [z_samples, batch_size]
        sum_log_p_yCz = sum_log_prob(p_yCc, Y_trgt)

        # E_{q(z|y_cntxt,y_trgt)}[...] . size = [batch_size]
        E_z_sum_log_p_yCz = sum_log_p_yCz.mean(0)

        # second term in loss is \sum_l KL[q(z^l|y_cntxt,y_trgt)||q(z^l|y_cntxt)]
        # KL[q(z^l|y_cntxt,y_trgt)||q(z^l|y_cntxt)]. size = [batch_size, *n_lat]
        kl_z = kl_divergence(q_zCct, q_zCc)
        # \sum_l ... . size = [batch_size]
        E_z_kl = sum_from_nth_dim(kl_z, 1)
        
        if R!=None:
            r_batch = R.shape[0]//2
            zis = R[:r_batch]
            zjs = R[r_batch:]

            if r_batch < self.batch_size*2:
                diff = self.batch_size - r_batch
                ids = np.random.randint(0,r_batch,diff)
                zis = torch.cat([zis,zis[ids]])
                zjs = torch.cat([zjs,zjs[ids]])

            nt_xent = nt_xent_criterion(zis, zjs)
            #print(-(E_z_sum_log_p_yCz - E_z_kl),nt_xent)
            return -(E_z_sum_log_p_yCz - E_z_kl)*self.lreg + nt_xent
        
        return -(E_z_sum_log_p_yCz - E_z_kl)

class NLLLossLNPF(BaseLossNPF):
    """
    Compute the approximate negative log likelihood for Neural Process family[?].

     Notes
    -----
    - might be high variance
    - biased
    - approximate because expectation over q(z|cntxt) instead of p(z|cntxt)
    - if q_zCct is not None then uses importance sampling (i.e. assumes that sampled from it).

    References
    ----------
    [?]
    """

    def get_loss(self, p_yCc, z_samples, q_zCc, q_zCct, Y_trgt):

        n_z_samples, batch_size, *n_trgt = p_yCc.batch_shape

        # computes approximate LL in a numerically stable way
        # LL = E_{q(z|y_cntxt)}[ \prod_t p(y^t|z)]
        # LL MC = log ( mean_z ( \prod_t p(y^t|z)) )
        # = log [ sum_z ( \prod_t p(y^t|z)) ] - log(n_z_samples)
        # = log [ sum_z ( exp \sum_t log p(y^t|z)) ] - log(n_z_samples)
        # = log_sum_exp_z ( \sum_t log p(y^t|z)) - log(n_z_samples)

        # \sum_t log p(y^t|z). size = [n_z_samples, batch_size]
        sum_log_p_yCz = sum_log_prob(p_yCc, Y_trgt)

        # uses importance sampling weights if necessary
        if q_zCct is not None:

            # All latents are treated as independent. size = [n_z_samples, batch_size]
            sum_log_q_zCc = sum_log_prob(q_zCc, z_samples)
            sum_log_q_zCct = sum_log_prob(q_zCct, z_samples)

            # importance sampling : multiply \prod_t p(y^t|z)) by q(z|y_cntxt) / q(z|y_cntxt, y_trgt)
            # i.e. add log q(z|y_cntxt) - log q(z|y_cntxt, y_trgt)
            sum_log_w_k = sum_log_p_yCz + sum_log_q_zCc - sum_log_q_zCct
        else:
            sum_log_w_k = sum_log_p_yCz

        # log_sum_exp_z ... . size = [batch_size]
        log_S_z_sum_p_yCz = torch.logsumexp(sum_log_w_k, 0)

        # - log(n_z_samples)
        log_E_z_sum_p_yCz = log_S_z_sum_p_yCz - math.log(n_z_samples)

        # NEGATIVE log likelihood
        return -log_E_z_sum_p_yCz


#! might need gradient clipping as in their paper
class SUMOLossLNPF(BaseLossNPF):
    """
    Estimate negative log likelihood for Neural Process family using SUMO [1].

    Notes
    -----
    - approximate because expectation over q(z|cntxt) instead of p(z|cntxt)
    - if q_zCct is not None then uses importance sampling (i.e. assumes that sampled from it).

    Parameters
    ----------
    p_n_z_samples : scipy.stats.rv_frozen, optional
        Distribution for the number of of z_samples to take.

    References
    ----------
    [1] Luo, Yucen, et al. "SUMO: Unbiased Estimation of Log Marginal Probability for Latent
    Variable Models." arXiv preprint arXiv:2004.00353 (2020)
    """

    def __init__(
        self,
        p_n_z_samples=LightTailPareto(a=5).freeze(85),
        **kwargs,
    ):
        super().__init__()
        self.p_n_z_samples = p_n_z_samples

    def get_loss(self, p_yCc, z_samples, q_zCc, q_zCct, Y_trgt):

        n_z_samples, batch_size, *n_trgt = p_yCc.batch_shape

        # \sum_t log p(y^t|z). size = [n_z_samples, batch_size]
        sum_log_p_yCz = sum_log_prob(p_yCc, Y_trgt)

        # uses importance sampling weights if necessary
        if q_zCct is not None:
            # All latents are treated as independent. size = [n_z_samples, batch_size]
            sum_log_q_zCc = sum_log_prob(q_zCc, z_samples)
            sum_log_q_zCct = sum_log_prob(q_zCct, z_samples)

            #! It should be p(y^t,z|cntxt) but we are using q(z|cntxt) instead of p(z|cntxt)
            # \sum_t log (q(y^t,z|cntxt) / q(z|cntxt,trgt)) . size = [n_z_samples, batch_size]
            sum_log_w_k = sum_log_p_yCz + sum_log_q_zCc - sum_log_q_zCct
        else:
            sum_log_w_k = sum_log_p_yCz

        # size = [n_z_samples, 1]
        ks = (torch.arange(n_z_samples) + 1).unsqueeze(-1)
        #! slow to always put on GPU
        log_ks = ks.float().log().to(sum_log_w_k.device)

        #! the algorithm in the paper is not correct on ks[:k+1] and forgot inv_weights[m:]
        # size = [n_z_samples, batch_size]
        cum_iwae = logcumsumexp(sum_log_w_k, 0) - log_ks

        #! slow to always put on GPU
        # you want reverse_cdf which is P(K >= k ) = 1 - P(K < k) = 1 - P(K <= k-1) = 1 - CDF(k-1)
        inv_weights = torch.from_numpy(1 - self.p_n_z_samples.cdf(ks - 1)).to(
            sum_log_w_k.device
        )

        m = self.p_n_z_samples.support()[0]
        # size = [batch_size]
        sumo = cum_iwae[m - 1] + (
            inv_weights[m:] * (cum_iwae[m:] - cum_iwae[m - 1 : -1])
        ).sum(0)

        nll = -sumo
        return nll
