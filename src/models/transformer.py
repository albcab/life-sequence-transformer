import torch
import torch.nn as nn
from torch.nn.utils import parametrize
from torch.nn.utils.rnn import pad_sequence
from src.models.embeddings import Embeddings
from src.models.transformer_utils import ScaleNorm, l2_norm, Center, Swish, gelu, gelu_new, swish, SigSoftmax
import logging

from transformers import (
  BertConfig,
  BigBirdConfig,
  EncoderDecoderConfig,
  EncoderDecoderModel
)
from src.models.modules import EncoderLayer, DecoderLayer


log = logging.getLogger(__name__)

ACT2FN = {
    "gelu": torch.nn.functional.gelu,
    "gelu_custom": gelu,
    "relu": torch.nn.functional.relu,
    "swish": swish,
    "gelu_google": gelu_new,
    "tanh": torch.tanh,
}


class Performer(nn.Module):
    def __init__(self, hparams, decoder=False, with_background=True):
        """Encoder part of the life2vec model (but with performer attention)"""
        super(Performer, self).__init__()

        hparams.is_decoder = decoder
        
        self.hparams = hparams
        # Initialize the Embedding Layer
        self.embedding = Embeddings(hparams=hparams, with_background=with_background)
        # Initialize the Encoder Blocks
        self.encoders = nn.ModuleList(
            [EncoderLayer(hparams) for _ in range(hparams.n_encoders)]
        )

    def forward(self, x, padding_mask):
        """Forward pass"""
        x, pos_emb = self.embedding(
            tokens=x[:, 0], year=x[:, 1], age=x[:, 2]
        )
        for layer in self.encoders:
            x = torch.einsum("bsh, bs -> bsh", x, padding_mask)
            x = layer(x, pos_emb=pos_emb, mask=padding_mask)
        return x

    def forward_finetuning(self, x, padding_mask=None):

        x, pos_emb = self.embedding(
            tokens=x[:, 0], year=x[:, 1], age=x[:, 2]
        )

        for _, layer in enumerate(self.encoders):
            x = torch.einsum("bsh, bs -> bsh", x, padding_mask)
            x = layer(x, pos_emb=pos_emb, mask=padding_mask)

        return x
    
    def forward_finetuning_cls(self, x, padding_mask):
        logits = list()
        x, pos_emb = self.embedding(
            tokens=x[:, 0], year=x[:, 1], age=x[:, 2]
        )
        for i, layer in enumerate(self.encoders):
            x = torch.einsum("bsh, bs -> bsh", x, padding_mask)
            x = layer(x, pos_emb=pos_emb, mask=padding_mask)
            if  i == (self.hparams.n_encoders - 1)//2 or i == 1 or i == (self.hparams.n_encoders - 1): ## we extract CLS embeddings after 0th and last encoder block and average those
                logits.append(x[:, 0])
        return x[:,0]
        return torch.stack(logits, dim=0).mean(dim=0)

    def forward_finetuning_with_embeddings(self, x, pos_emb, padding_mask):
        ### Inputs are the embeddings (not sequence of tokens)
        for _, layer in enumerate(self.encoders):
            x = torch.einsum("bsh, bs -> bsh", x, padding_mask)
            x = layer(x, pos_emb=pos_emb, mask=padding_mask)
        return x

    def forward_finetuning_with_embeddings_cls(self, x, pos_emb, padding_mask):
        ### Inputs are the embeddings (not sequence of tokens)
        logits = list()
        for i, layer in enumerate(self.encoders):
            x = torch.einsum("bsh, bs -> bsh", x, padding_mask)
            x = layer(x, pos_emb=pos_emb, mask=padding_mask)
            if  i == (self.hparams.n_encoders - 1)//2 or i == 1 or i == (self.hparams.n_encoders - 1): ## we extract CLS embeddings after 0th and last encoder block and average those
                logits.append(x[:, 0])
        return torch.stack(logits, dim=0).mean(dim=0)

    def get_sequence_embedding(self, x):
        """Get only embeddings"""
        return self.embedding(
            tokens=x[:, 0], year=x[:, 1], age=x[:, 2]
        )

    def redraw_projection_matrix(self, batch_idx: int):
        """Redraw projection Matrices for each layer (only valid for Performer)"""
        if batch_idx == -1:
            log.info("Redrawing projections for the encoder layers (manually)")
            for encoder in self.encoders:
                encoder.redraw_projection_matrix()

        elif batch_idx > 0 and batch_idx % self.hparams.feature_redraw_interval == 0:
            log.info("Redrawing projections for the encoder layers")
            for encoder in self.encoders:
                encoder.redraw_projection_matrix()


class Transformer(nn.Module):
    def __init__(self, hparams, num_cross_decoder=2, with_background=False):
        """Encoder-Decoder version of the life2vec model"""
        super(Transformer, self).__init__()

        hparams.is_decoder = True
        self.hparams = hparams
        self.num_cross_decoder = num_cross_decoder
        # Initialize the Embedding Layer
        self.embedding = Embeddings(hparams=hparams, with_background=with_background)
        # Initialize the Decoder
        self.decoders = nn.ModuleList(
            [DecoderLayer(hparams) for _ in range(num_cross_decoder)] +
            [EncoderLayer(hparams) for _ in range(hparams.n_decoders - num_cross_decoder)]
        )
        # Initialize the Encoder (change hparams to encoder hparams, bit hacky but dont want to change Perfomer code)
        hparams.hidden_ff = hparams.encoder_hidden_ff
        hparams.hidden_act = hparams.encoder_hidden_act
        hparams.n_heads = hparams.encoder_n_heads
        hparams.n_local = hparams.encoder_n_local
        hparams.local_window_size = hparams.encoder_local_window_size
        hparams.num_random_features = hparams.encoder_num_random_features
        hparams.is_decoder = False
        self.encoders = nn.ModuleList(
            [EncoderLayer(hparams) for _ in range(hparams.n_encoders)]
        )
        true_tensor = torch.tensor([True])
        background_padding_mask = true_tensor.repeat(hparams.batch_size, 4)
        self.register_buffer("background_padding_mask", background_padding_mask, persistent=False)

    def encode(self, z):
        z = self.embedding.forward_indep(z['tokens'].long())#, z['year'].long(), z['month'].long())
        for layer in self.encoders:
            z = layer(z, mask=self.background_padding_mask)
        return z

    def decode(self, x, encoder_hidden_states=None, padding_mask=None):

        # Shape of x_emb: (batch_size, seq_len, d_model)
        x, pos_emb = self.embedding(tokens=x[:, 0], year=x[:, 1], age=x[:, 2])

        for i, layer in enumerate(self.decoders):
            x = torch.einsum("bsh, bs -> bsh", x, padding_mask)
            if i < self.num_cross_decoder:
                x = layer(x, context=encoder_hidden_states, mask=padding_mask, pos_emb=pos_emb)
            else:
                x = layer(x, mask=padding_mask, pos_emb=pos_emb)

        # Shape of logits: (batch_size, seq_len, tuple_size, vocab_size)
        return x


    def forward(self, x, z=None, padding_mask=None):
        encoder_hidden = self.encode(z)

        out = self.decode(x, encoder_hidden_states=encoder_hidden, padding_mask=padding_mask)

        return out
    
    def forward_bol(self, x, z, padding_mask=None):
        encoder_hidden = self.encode(z)

        x = self.decode(x, encoder_hidden_states=encoder_hidden, padding_mask=padding_mask)
        return x[:,0]

    # def forward_finetuning_with_embeddings(self, x, padding_mask):
    #     ### Inputs are the embeddings (not sequence of tokens)
    #     for _, layer in enumerate(self.encoders):
    #         x = torch.einsum("bsh, bs -> bsh", x, padding_mask)
    #         x = layer(x, padding_mask)
    #     return x

    # def forward_finetuning_with_embeddings_cls(self, x, padding_mask):
    #     ### Inputs are the embeddings (not sequence of tokens)
    #     logits = list()
    #     for i, layer in enumerate(self.encoders):
    #         x = torch.einsum("bsh, bs -> bsh", x, padding_mask)
    #         x = layer(x, padding_mask)
    #         if  i == (self.hparams.n_encoders - 1)//2 or i == 1 or i == (self.hparams.n_encoders - 1): ## we extract CLS embeddings after 0th and last encoder block and average those
    #             logits.append(x[:, 0])
    #     return torch.stack(logits, dim=0).mean(dim=0)

    def get_sequence_embedding(self, x):
        """Get only embeddings"""
        return self.embedding(
            tokens=x[:, 0], year=x[:, 1], age=x[:, 2]
        )

    def redraw_projection_matrix(self, batch_idx: int):
        """Redraw projection Matrices for each layer (only valid for Performer)"""
        if batch_idx == -1:
            log.info("Redrawing projections for the encoder and decoder layers (manually)")
            for encoder in self.encoders:
                encoder.redraw_projection_matrix()
            for decoder in self.decoders:
                decoder.redraw_projection_matrix()

        elif batch_idx > 0 and batch_idx % self.hparams.feature_redraw_interval == 0:
            log.info("Redrawing projections for the encoder and decoder layers")
            for encoder in self.encoders:
                encoder.redraw_projection_matrix()
            for decoder in self.decoders:
                decoder.redraw_projection_matrix()


class MaskedLanguageModel(nn.Module):
    """Masked Language Model (MLM) Decoder (for pretraining)"""

    def __init__(self, hparams, embedding, act: str = "tanh", with_background=True):
        super(MaskedLanguageModel, self).__init__()
        self.hparams = hparams
        self.act = ACT2FN[act]
        self.dropout = nn.Dropout(p=self.hparams.emb_dropout)

        self.V = nn.Linear(self.hparams.hidden_size, self.hparams.hidden_size)
        self.g = nn.Parameter(torch.tensor([hparams.hidden_size**0.5]))
        self.out = nn.Linear(
            self.hparams.hidden_size,
            self.hparams.vocab_size,
            bias=False
        )
        if self.hparams.weight_tying == "wt":
            log.info("MLM decoder WITH Wight Tying")
            try:
                self.out.weight = embedding.token.parametrizations.weight.original
            except:
                log.warning("MLM decoder parametrization failed")
                self.out.weight = embedding.token.weight

        if self.hparams.parametrize_emb:
            if with_background:
                ignore_index = torch.LongTensor([0, 6, 7, 8, 9])
            else:
                ignore_index = torch.LongTensor([0, 5, 6, 7, 8, 9])
            log.info("(MLM Decoder) centering: true normalisation: %s" %
                     hparams.norm_output_emb)
            parametrize.register_parametrization(self.out, "weight", Center(
                ignore_index=ignore_index, norm=hparams.norm_output_emb))

    def batched_index_select(self, x, dim, indx):
        """Gather the embeddings of tokens that we should make prediction on"""
        indx_ = indx.unsqueeze(2).expand(
            indx.size(0), indx.size(1), x.size(-1))
        return x.gather(dim, indx_)

    def forward(self, logits, batch):
        indx = batch["target_pos"].long()
        logits = self.dropout(self.batched_index_select(logits, 1, indx))
        logits = self.dropout(l2_norm(self.act(self.V(logits))))
        return self.g * self.out(logits)


class SOP_Decoder(nn.Module):
    """Sequence Order Decoder (for pretraining)"""

    def __init__(self, hparams):
        super(SOP_Decoder, self).__init__()
        hidden_size = hparams.hidden_size
        num_targs = hparams.cls_num_targs
        p = hparams.dc_dropout

        self.in_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=p)
        self.norm = ScaleNorm(hidden_size=hidden_size, eps=hparams.epsilon)

        self.act = ACT2FN["swish"]
        self.out_layer = nn.Linear(hidden_size, num_targs)

    def forward(self, x, **kwargs):
        """Foraward Pass"""
        x = self.dropout(self.norm(self.act(self.in_layer(x))))
        return self.out_layer(x)


class CLS_Decoder(nn.Module):
    """Classification for CLS Predictions"""
    def __init__(self, hparams):
        super(CLS_Decoder, self).__init__()
        hidden_size = hparams.hidden_size
        num_targs = hparams.num_targets
        p = hparams.dc_dropout

        self.ff1 = nn.Linear(hidden_size,hidden_size)
        self.dropout = nn.Dropout(p=p)
        self.norm_1 = ScaleNorm(hidden_size=hidden_size, eps=hparams.epsilon)

        self.act_1 = Swish()#nn.Tanh()
        self.out_layer = nn.Linear(hidden_size, num_targs)

    def forward(self, x, **kwargs):
        """Foraward Pass"""  
        #x = self.norm_1(self.act_1(self.ff1(self.dropout(x))))
        x = self.dropout(self.norm_1(self.act_1(self.ff1(x))))
        return self.out_layer(x)


class AttentionDecoder(nn.Module):

    def __init__(self, hparams, num_outputs:int) -> None:
        super().__init__()
        hidden_size = hparams.hidden_size
        context_size = hidden_size // 2
        ## LAYERS
        self.ff = nn.Linear(hidden_size, hidden_size)
        self.pool = nn.Linear(hidden_size, context_size)
        #self.post = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, num_outputs)
        ## ACTIVATIONS
        self.act = Swish()
        self.tanh = nn.Tanh()
        self.sigsoftmax = SigSoftmax(dim = -1)
        ## MODULES
        self.dropout = nn.AlphaDropout(p=hparams.dc_dropout)
        self.norm = ScaleNorm(hidden_size=hidden_size, eps=hparams.epsilon)

        self.identity = nn.Identity()

        ## CONTEXT VECTOR
        self.register_parameter(name="context", param=nn.Parameter(
                torch.randn(context_size)))

        self.attn = None

    def attention_pooling(self, x, mask):
        h = self.tanh(self.pool(x))
        scores = torch.mul(h, self.context) \
                      .sum(dim = -1, keepdim=False)
        scores = self.sigsoftmax(scores, mask.bool())
        self.attn = scores
        return  torch.mul(x, scores.unsqueeze(-1)).sum(dim = 1)

    def forward(self, x, mask):
        logits = self.dropout(self.norm(self.act(self.ff(x))))
        logits = self.identity(self.attention_pooling(x = logits, mask = mask))
        return self.out(logits)


class NextTokenDecoder(nn.Module):
    """Next Token Decoder for last layer of EncDec or decoder only"""

    def __init__(self, hparams, embedding, with_background=True):
        super(NextTokenDecoder, self).__init__()
        self.hparams = hparams
        self.out = nn.Linear(
            self.hparams.hidden_size,
            self.hparams.vocab_size,
            bias=False
        )
        if self.hparams.weight_tying == "wt":
            log.info("MLM decoder WITH Wight Tying")
            try:
                self.out.weight = embedding.token.parametrizations.weight.original
            except:
                log.warning("MLM decoder parametrization failed")
                self.out.weight = embedding.token.weight

        if self.hparams.parametrize_emb:
            if with_background:
                ignore_index = torch.LongTensor([0, 6, 7, 8, 9])
            else:
                ignore_index = torch.LongTensor([0, 5, 6, 7, 8, 9])
            log.info("Decoder centering: true normalisation: %s" %
                     hparams.norm_output_emb)
            parametrize.register_parametrization(self.out, "weight", Center(
                ignore_index=ignore_index, norm=hparams.norm_output_emb))

    def forward(self, logits):
        return self.out(logits)
