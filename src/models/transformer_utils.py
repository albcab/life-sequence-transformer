import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import logging
import torch
from torch import nn
import numpy as np

log = logging.getLogger(__name__)


def masked_sort(x, mask, dim=-1, descending=True):
    """Sort operation with the masked elements"""
    mask = mask.type(x.dtype)
    masked = torch.mul(x, mask)
    neg_inf = torch.zeros_like(x).to(
        masked.device).masked_fill(mask == 0, -math.inf)
    return torch.sort((masked + neg_inf), dim=dim, descending=descending)[0]


def masked_max(x, mask, dim=-1):
    """Max operation with the masked elements"""
    mask = mask.type(x.dtype)
    masked = torch.mul(x, mask)
    neg_inf = torch.zeros_like(x).to(
        masked.device).masked_fill(mask == 0, -math.inf)
    return torch.max((masked + neg_inf), dim=1)[0]


class Center(nn.Module):
    """Remove the mean from the embedding matrix"""

    def __init__(self,  ignore_index: torch.LongTensor, norm: bool = False, use_ignore_index: bool = True) -> None:
        super().__init__()
        self.register_buffer("norm", torch.BoolTensor([norm]))
        self.register_buffer("use_ignore_index",
                             torch.BoolTensor([use_ignore_index]))

        self.register_buffer("ignore_index", ignore_index)

    def forward(self, X):
        if self.use_ignore_index:
            mask = self.mask(X)
            # we do not want tokens as PAD and PLACEHOLDERS to contribute to the mean
            X = X - X[mask].mean(0)
            # we do not want to do anything with the indexes we ignore, like PAD and PLACEHOLDERS
            X[self.ignore_index] *= 0
        else:
            X = X - X.mean(0)
        if self.norm:
            return l2_norm(X)
        return X

    def mask(self, X):
        mask = torch.ones(X.shape[0])
        mask[self.ignore_index] = 0
        return mask.bool()


def cosine_annealing(current_step):
    """Cosine Annealing for the Learning Rate"""
    progress = min(current_step * 0.033, 0.95)
    return math.cos(0.5 * math.pi * progress)


#######################
# Activation Functions
#######################

def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """ ""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
    Also see https://arxiv.org/abs/1606.08415
    """ ""
    return (
        0.5
        * x
        * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    )


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {
    "gelu": torch.nn.functional.gelu,
    "gelu_custom": gelu,
    "relu": torch.nn.functional.relu,
    "swish": swish,
    "gelu_google": gelu_new,
}

###################################
# Normalisation and Residuals
###################################


class Swish(nn.Module):
    def forward(self, input: Tensor):
        return swish(input)


def l2_norm(x):
    return F.normalize(x, dim=-1, p=2)


class Norm(nn.Module):
    def forward(self, X):
        return l2_norm(X)


class ReZero(torch.nn.Module):
    """Implementation of ReZero (Residual Connection)"""

    def __init__(self, hidden_size, simple: bool = True, fill: float = .0):
        """"""
        super(ReZero, self).__init__()
        if simple:  # aka original
            self.weights = torch.nn.Parameter(torch.add(torch.zeros(1), fill))
        else:
            self.weights = torch.nn.Parameter(
                torch.add(torch.zeros(hidden_size), fill))

    def forward(self, x, y):
        return x + y * self.weights


class ScaleNorm(torch.nn.Module):
    """L2-norm (Alternative to LayerNorm)"""

    def __init__(self, hidden_size, eps=1e-6):
        """"""
        super(ScaleNorm, self).__init__()
        self.g = torch.nn.Parameter(torch.sqrt(torch.Tensor([hidden_size])))
        self.eps = eps

    def forward(self, x):
        """"""
        norm = self.g / torch.linalg.norm(x, dim=-1, ord=2, keepdim=True).clamp(
            min=self.eps
        )
        return x * norm


class FixNorm(ScaleNorm):
    """Scale Norm with fixed weight (no gradient)"""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__(hidden_size, eps)
        self.g.weight = torch.Tensor([1.])
        self.g.requires_grad = False


class SigSoftmax(nn.Module):
    """Implementation of SigSoftmax (prevents oversaturation)"""

    def __init__(self, dim: int = -1, epsilon: float = 1e-12):
        super().__init__()
        self.epsilon = epsilon
        self.sigmoid = nn.LogSigmoid()
        self.softmax = nn.Softmax(dim)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x.masked_fill(~mask, -np.inf)
        return self.softmax(x + torch.log(torch.sigmoid(x) + self.epsilon))


###############
## Loss Fn
###############

class AsymmetricCrossEntropyLoss(nn.Module):
    """CrossEntropy Loss for Positive-Unlabeled Learning
    Args:
    """
    def __init__(self, pos_weight: float = 0.5, penalty: float = 0., sigmoid: bool = False):
        super().__init__()
        #self.softmax = nn.Softmax(dim=1)
        #self.loss_positive = nn.CrossEntropyLoss(ignore_index=0 , reduction="sum")
        #self.loss_negative = nn.CrossEntropyLoss(ignore_index=1, reduction="sum")
        self.sigmoid = sigmoid
        if self.sigmoid:
            self.ls = nn.LogSigmoid()
        else:
            self.ls= nn.LogSoftmax(dim = 1)
        self.loss = nn.NLLLoss()
        self.register_buffer("penalty", torch.tensor([penalty, 0.]))
        self.register_buffer("weight", torch.tensor([1-pos_weight, pos_weight]))

    def __calculate_loss__(self, loss_u, loss_p, n_u, n_p):
        loss_u = loss_u * self.weight[0]
        loss_p = loss_p * self.weight[1]
        if n_p == 0: 
            return loss_u
        elif n_u == 0:
            return loss_p
        else:
            return loss_p + loss_u
    
    def set_penalty(self, penalty):
        self.penalty[0] = penalty
    
    def adjust_penalty(self):
        self.penalty[0] = torch.addself.penalty[0] * 0.9 + 0.1

    @property
    def biased_penalty(self):
        """Penalty for ACE loss without SCAR assumption"""
        penalty = self.penalty 
        penalty[0] = penalty[0] / (1.0 - penalty[0])
        return penalty

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """
        Args:
        logits: raw logits (batch_size, num_classes)
        targets: one hot encoded (batch_size, num_classes)
        """
        n_p = target[:,1].sum()
        n_u = target.shape[0] - n_p
        is_positive, is_negative = (target[:,1] == 1).type(logits.dtype), (target[:,0] == 1).type(logits.dtype)

        if self.sigmoid:
            ## logits +=1
            return F.binary_cross_entropy_with_logits(logits.squeeze(), target=target[:,1].squeeze().float(), reduction="mean")
        scores = self.ls(logits)
        loss_p =  F.nll_loss(scores, target = target[:,1], ignore_index = 0, reduction="mean")
            #scores = logits + self.penalty.expand(scores.shape[0], 2)
        scores = self.ls(logits)
        loss_u = F.nll_loss(scores, target = target[:,1], ignore_index=1, reduction="mean")
        return self.__calculate_loss__(loss_u=loss_u,
                                       loss_p=loss_p,
                                       n_p=n_p, n_u = n_u)

        #else: ## without SCAR assumption
        #    scores = self.ls(logits + self.biased_penalty)
