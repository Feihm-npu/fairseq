# mixtral.py
import math
import torch
import torch.nn as nn
from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqDecoder
from .mixtral_layer import MixtralDecoderLayer, MixtralRMSNorm

DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)

def fsdp_wrap_expert(args, layer, min_num_params=0):

    process_group = layer.moe_layer.expert_group
    for i, expert in enumerate(layer.moe_layer.experts):
        layer.moe_layer.experts[i] = fsdp_wrap(expert, process_group=process_group, min_num_params=0)
    layer = fsdp_wrap(layer, min_num_params=min_num_params)
    return layer

def build_mixtral_decoder(args, task):
    dictionary = task.target_dictionary
    embed_tokens = nn.Embedding(len(dictionary), args.decoder_embed_dim, dictionary.pad())
    decoder = MixtralDecoder(args, dictionary, embed_tokens, no_encoder_attn=True)
    return decoder

class MixtralDecoder(FairseqDecoder):


    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(dictionary)
        self.args = args
        self.register_buffer("version", torch.Tensor([1]))

        self.dropout_module = nn.Dropout(args.dropout)
        self.embed_tokens = embed_tokens
        self.padding_idx = embed_tokens.padding_idx
        self.embed_dim = args.decoder_embed_dim
        self.max_target_positions = args.max_position_embeddings


        self.embed_scale = 1.0 if getattr(args, "no_scale_embedding", False) else math.sqrt(self.embed_dim)

        self.final_layer_norm = MixtralRMSNorm(self.embed_dim, eps=1e-6)

        self.layers = nn.ModuleList([])
        for i in range(args.decoder_layers):
            layer = MixtralDecoderLayer(args)
            # activation checkpoint
            if getattr(args, "checkpoint_activations", False):
                from fairseq.modules.checkpoint_activations import checkpoint_wrapper
                offload_to_cpu = getattr(args, "offload_activations", False)
                layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)

            # 根据是否是MoE层选择fsdp wrap函数
            if True:
                layer = fsdp_wrap_expert(args, layer, min_num_params=0)
            else:
                min_params_to_wrap = getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
                layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)

            self.layers.append(layer)

    def forward(self, src_tokens, incremental_state=None, **kwargs):
        x = self.embed_tokens(src_tokens)
        x = self.embed_scale * x
        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = [x]
        l_aux = []

        for layer in self.layers:
            x, layer_l_aux = layer(
                x,
                incremental_state=incremental_state,
            )
            if layer_l_aux is not None:
                l_aux.extend(layer_l_aux)
            inner_states.append(x)

        x = self.final_layer_norm(x)
        # T x B x C -> B x T x C
        return x.transpose(0,1), {"l_aux": l_aux, "inner_states": inner_states}

    def max_positions(self):
        return self.max_target_positions
