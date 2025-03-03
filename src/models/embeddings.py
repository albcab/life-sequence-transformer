import torch
import torch.nn as nn
from torch.nn.utils import parametrize
from src.models.transformer_utils import ReZero, Center

import logging
log = logging.getLogger(__name__)


class Embeddings(nn.Module):
    """Class for token, position, segment and backgound embedding."""

    def __init__(self, hparams, with_background=False):
        super(Embeddings, self).__init__()
        embedding_size = hparams.hidden_size
        self.with_background = with_background

        # Initialize Token/Concept embedding matrix
        self.token = nn.Embedding(
            hparams.vocab_size,
            embedding_size,
            padding_idx=0
        )

        # Initialize Time2Vec embeddings
        ###MIGHT WANT TO CONSIDER USING BOTH COS AND SIN, SPECIALLY FOR MONTHS
        # self.month = PositionalEmbedding(1, hparams.hidden_size, torch.sin)
        self.age = PositionalEmbedding(1, hparams.hidden_size, torch.cos)
        self.year = PositionalEmbedding(1, hparams.hidden_size, torch.sin)

        # Uniformly initialise the weights of the embedding matrix
        d = 0.01
        nn.init.uniform_(self.token.weight, a=-d, b=d)

        if hparams.parametrize_emb:
            try:
                # The centering of the embedding matrix
                self.parametrize(norm=hparams.norm_input_emb)
            except:
                log.info(
                    "(EMBEDDING) Normalisation hyperparameter is not found, set to FALSE")
                self.parametrize()

        #COULD JUST ADD MEBEDDINGS INSTEAD OF LEARNING PARAMETERS FOR MIXING
        self.res_age = ReZero(hparams.hidden_size, simple=True, fill=0)
        # # self.res_age = lambda t, m: t + m
        self.res_year = ReZero(hparams.hidden_size, simple=True, fill=0)
        # self.res_year = lambda t, y: t + y
        self.dropout = nn.Dropout(hparams.emb_dropout)

    def parametrize(self, norm: bool = False):
        """Remove Mean from the Embedding Matrix (on each forward pass"""
        if self.with_background:
            ignore_index = torch.LongTensor(
                [0, 6, 7, 8, 9]) ###USE PLACEHOLDER0 AS CLS/BOS FOR SOP IN MODEL WITH BACKGROUND
        else:
            ignore_index = torch.LongTensor(
                [0, 5, 6, 7, 8, 9])  # We ignore the 5 tokens (PAD, and placeholder tokens that we included into the language but never used)

        parametrize.register_parametrization(
            self.token, "weight", Center(ignore_index=ignore_index, norm=norm))

    def reparametrization(self):
        """Remove the parametrization from the Concept Embedding Matrix"""

        parametrize.remove_parametrizations(
            self.token, "weight", leave_parametrized=False)

    def forward(self, tokens, year, age):
        """"""
        tokens = self.token(tokens)

        pos = self.year(year.float().unsqueeze(-1))
        if self.with_background:
            pos[:, :4] *= 0
        else:
            pos[:, :1] *= 0
        tokens = self.res_year(tokens, pos)

        pos = self.age(age.float().unsqueeze(-1))
        if self.with_background:
            pos[:, :4] *= 0
        else:
            pos[:, :1] *= 0
        tokens = self.res_age(tokens, pos)

        return self.dropout(tokens), None
    
    def forward_indep(self, tokens):#, year, month):
        """"""
        ###DROPOUT ONLY ON TOKENS
        return self.dropout(self.token(tokens))
        year = self.year(year.float().unsqueeze(-1)).unsqueeze(1)
        month = self.token(month.unsqueeze(-1))
        # month = self.month(month.float().unsqueeze(-1)).unsqueeze(1)
        # indx = torch.randperm(4) #ordering of background should not matter, treat all permutations of the tokens equally

        return torch.cat((tokens, year, month), dim=1)#[:, indx, :]


# TIME2VEC IMPLEMENTATION

def t2v(tau, f, w, b, w0, b0, arg=None):
    """Time2Vec function"""
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], -1)


class PositionalEmbedding(nn.Module):
    """Implementation of Time2Vec"""

    def __init__(self, in_features, out_features, f):
        super(PositionalEmbedding, self).__init__()

        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(
            torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(
            torch.randn(in_features, out_features - 1))
        self.f = f

        d = 0.01
        nn.init.uniform_(self.w0, a=-d, b=d)
        nn.init.uniform_(self.b0, a=-d, b=d)
        nn.init.uniform_(self.w, a=-d, b=d)
        nn.init.uniform_(self.b, a=-d, b=d)

    def forward(self, tau):
        return t2v(tau, self.f, self.w, self.b, self.w0, self.b0)
