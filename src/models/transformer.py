import torch
import torch.nn as nn
from torch.nn.utils import parametrize
from torch.nn.utils.rnn import pad_sequence
from src.models.embeddings import Embeddings
from src.models.transformer_utils import ScaleNorm, l2_norm, Center, Swish, gelu, gelu_new, swish, SigSoftmax
import logging

from transformers import (
  BertConfig,
  EncoderDecoderConfig,
  EncoderDecoderModel
)


log = logging.getLogger(__name__)

ACT2FN = {
    "gelu": torch.nn.functional.gelu,
    "gelu_custom": gelu,
    "relu": torch.nn.functional.relu,
    "swish": swish,
    "gelu_google": gelu_new,
    "tanh": torch.tanh,
}


class Transformer(nn.Module):
    def __init__(self, hparams, decoder=False):
        """Encoder part of the life2vec model"""
        super(Transformer, self).__init__()

        self.hparams = hparams
        # Initialize the Embedding Layer
        self.embedding = Embeddings(hparams=hparams)
        # Initialize the Encoder-Encoder Transformer
        encoder_config = BertConfig(
            vocab_size=1,
            pad_token_id=0,
            hidden_size=hparams.hidden_size,
            num_hidden_layers=hparams.encoder_layers,
            num_attention_heads=hparams.encoder_attention_heads,
            intermediate_size=hparams.intermediate_size,
            # hidden_act=ACT2FN[hparams.hidden_act],
            hidden_act=hparams.hidden_act,
            hidden_dropout_prob=hparams.fw_dropout,
            attention_probs_dropout_prob=hparams.att_dropout,
            max_position_embeddings=hparams.max_length,
            position_embedding_type=None #Reflects the unordered nature of the tokens,Simpler architecture,Focus on the content
        )
        decoder_config = BertConfig(
            vocab_size=1,
            pad_token_id=0,
            hidden_size=hparams.hidden_size,
            num_hidden_layers=hparams.decoder_layers,
            num_attention_heads=hparams.decoder_attention_heads,
            intermediate_size=hparams.intermediate_size,
            # hidden_act=ACT2FN[hparams.hidden_act],
            hidden_act=hparams.hidden_act,
            hidden_dropout_prob=hparams.dc_dropout,
            attention_probs_dropout_prob=hparams.att_dropout,
            max_position_embeddings=hparams.max_length,
            position_embedding_type='relative_key_query',
            is_decoder=decoder,
        )
        config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
        self.transformer = EncoderDecoderModel(config)
        self.transformer.config.decoder.is_decoder = decoder #should be enough for an encoder-decoder model?
        self.is_decoder = decoder
        self.transformer.config.decoder.add_cross_attention = True

        self.out_layer = nn.Linear(hparams.hidden_size, hparams.vocab_size, bias=False)

    def encode(self, z):
        z_emb = self.embedding.forward_indep(z['tokens'].long())#, z['year'].long(), z['month'].long())
        out = self.transformer.encoder(inputs_embeds=z_emb, output_hidden_states=True)
        encoder_hidden = out.hidden_states[-1]
        return encoder_hidden

    def decode(self, x, encoder_hidden_states=None, decoder_attention_mask=None):
        seq_len = x.size(2)

        # Shape of x_emb: (batch_size, seq_len, d_model)
        x_emb, _ = self.embedding(tokens=x[:, 0], year=x[:, 1], age=x[:, 2])

        # # Add latent embedding to input embeddings
        # if bar_ids is not None:
        #   assert bar_ids.max() <= encoder_hidden.size(1)
        #   embs = torch.cat([torch.zeros(x.size(0), 1, self.d_model, device=self.device), encoder_hidden], dim=1)
        #   offset = (seq_len * torch.arange(bar_ids.size(0), device=self.device)).unsqueeze(1)
        #   # Use bar_ids to gather encoder hidden states s.t. latent_emb[i, j] == encoder_hidden[i, bar_ids[i, j]]
        #   latent_emb = F.embedding((bar_ids + offset).view(-1), embs.view(-1, self.d_model)).view(x_emb.shape)
        #   x_emb += latent_emb

        if encoder_hidden_states is not None:
            # Make x_emb and encoder_hidden_states match in sequence length. Necessary for relative positional embeddings
            padded = pad_sequence([x_emb.transpose(0, 1), encoder_hidden_states.transpose(0, 1)], batch_first=True)
            x_emb, encoder_hidden_states = padded.transpose(1, 2)

            if self.is_decoder:
                out = self.transformer.decoder(
                    inputs_embeds=x_emb, 
                    encoder_hidden_states=encoder_hidden_states, 
                    output_hidden_states=True
                )
            else:
                out = self.transformer.decoder(
                    inputs_embeds=x_emb, 
                    encoder_hidden_states=encoder_hidden_states, 
                    output_hidden_states=True,
                    attention_mask=decoder_attention_mask
                )
            hidden = out.hidden_states[-1][:, :seq_len]
        else:
            out = self.transformer.decoder(inputs_embeds=x_emb, output_hidden_states=True)
            hidden = out.hidden_states[-1][:, :seq_len]

        # Shape of logits: (batch_size, seq_len, tuple_size, vocab_size)

        if not self.is_decoder:
            return hidden
        else:
            return self.out_layer(hidden)


    def forward(self, x, z=None, decoder_attention_mask=None):
        encoder_hidden = self.encode(z)

        out = self.decode(x, encoder_hidden_states=encoder_hidden, decoder_attention_mask=decoder_attention_mask)

        return out
    
    def forward_bol(self, x, z, decoder_attention_mask=None):
        encoder_hidden = self.encode(z)

        x = self.decode(x, encoder_hidden_states=encoder_hidden, decoder_attention_mask=decoder_attention_mask)
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

    # def redraw_projection_matrix(self, batch_idx: int):
    #     """Redraw projection Matrices for each layer (only valid for Performer)"""
    #     if batch_idx == -1:
    #         log.info("Redrawing projections for the encoder layers (manually)")
    #         for encoder in self.encoders:
    #             encoder.redraw_projection_matrix()

    #     elif batch_idx > 0 and batch_idx % self.hparams.feature_redraw_interval == 0:
    #         log.info("Redrawing projections for the encoder layers")
    #         for encoder in self.encoders:
    #             encoder.redraw_projection_matrix()


class MaskedLanguageModel(nn.Module):
    """Masked Language Model (MLM) Decoder (for pretraining)"""

    def __init__(self, hparams, embedding, act: str = "tanh"):
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
