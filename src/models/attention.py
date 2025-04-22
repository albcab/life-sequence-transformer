import torch.nn as nn
from src.models.performer import CustomSelfAttention, CustomCrossAttention
import logging

log = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """Multi Head Attention Module (only support Performer Attention but you can add any)"""

    def __init__(self, hparams):
        super(MultiHeadAttention, self).__init__()

        assert (
            hparams.hidden_size % hparams.n_heads == 0
        ), "Incorrect size of latent space or number of heads"
        assert hparams.attention_type in [
            "full",
            "multi_block_sparse",
            "simulated_sparse",
            "fast",
            "performer",
        ], NotImplementedError("Wrong Attention Type")

        self.head_size = hparams.hidden_size // hparams.n_heads
        self.head_num = hparams.n_heads
        self.attention_type = hparams.attention_type

        if self.attention_type == "full":
            raise NotImplementedError("Full Softmax Attention is deprecated")
        elif self.attention_type == "multi_block_sparse":
            """BigBird Type Attention"""
            raise NotImplementedError(
                "BigBird Attention is deprecated since v8")
        elif self.attention_type == "performer":
            """Performer Type Attention with the Local Attention Heads"""
            try:
                dropout = hparams.att_dropout
            except:
                print("Could not find dropout_attention parameter")
                dropout = 1e-3

            self.attention = CustomSelfAttention(
                dim=hparams.hidden_size,
                heads=self.head_num,
                nb_features=hparams.num_random_features,
                dim_head=hparams.hidden_size // self.head_num,
                causal=hparams.is_decoder,
                generalized_attention=False,
                no_projection=False,
                dropout=dropout,
                local_heads=hparams.n_local,
                qkv_bias=False,
                attn_out_bias=True,
                local_window_size=hparams.local_window_size,
            )

        else:
            raise NameError("Unknown attention type")

        self.step = 0

    def forward(self, x, mask=None, pos_emb=None):
        if self.attention_type == "performer":
            """
            FAVOR+ with the Local Heads (aka Performer Attention)
            """
            mask = mask.bool()
            out = self.attention(
                x, pos_emb=pos_emb, mask=mask, context_mask=mask.bool(), pos=None
            )
            return out
        else:
            raise NotImplementedError("Unknown Attention Type")


class MultiHeadCrossAttention(nn.Module):
    """Multi Head Cross Attention Module (only support Performer Attention but you can add any)"""

    def __init__(self, hparams):
        super(MultiHeadCrossAttention, self).__init__()

        assert (
            hparams.hidden_size % hparams.n_heads == 0
        ), "Incorrect size of latent space or number of heads"
        assert hparams.attention_type in [
            "performer",
        ], NotImplementedError("Wrong Attention Type")

        self.head_size = hparams.hidden_size // hparams.n_heads
        self.head_num = hparams.n_heads
        self.attention_type = hparams.attention_type

        if self.attention_type == "performer":
            """Performer Type Attention with the Local Attention Heads"""
            try:
                dropout = hparams.att_dropout
            except:
                print("Could not find dropout_attention parameter")
                dropout = 1e-3

            self.attention = CustomCrossAttention(
                dim=hparams.hidden_size,
                heads=self.head_num,
                nb_features=hparams.num_random_features,
                dim_head=hparams.hidden_size // self.head_num,
                causal=False,
                generalized_attention=False,
                no_projection=False,
                dropout=dropout,
                local_heads=0,
                qkv_bias=False,
                attn_out_bias=True,
                local_window_size=256,
            )

        else:
            raise NameError("Unknown attention type")

        self.step = 0

    def forward(self, x, context, mask=None, context_mask=None):
        if self.attention_type == "performer":
            """
            FAVOR+ with the Local Heads (aka Performer Attention)
            """
            mask = mask.bool()
            out = self.attention(
                x, context=context, mask=mask, context_mask=context_mask, pos=None
            )
            return out
        else:
            raise NotImplementedError("Unknown Attention Type")
