# mixtral_lm.py
import logging
from dataclasses import dataclass, field
from typing import Optional
from fairseq.dataclass import FairseqDataclass

from fairseq.tasks.language_modeling import LanguageModelingConfig
from fairseq.models import (
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from transformers import MixtralConfig 
import math

logger = logging.getLogger(__name__)

@dataclass
class MixtralLanguageModelingConfig(FairseqDataclass):
    hf_model_name: str = field(
        default="mistralai/Mixtral-8x7B-v0.1",
        metadata={"help": "Name or path of the Hugging Face Mixtral model and tokenizer"}
    )
    moe_expert_count: int = field(default=8, metadata={"help": "Number of experts"})
    moe_second_expert_policy: str = field(default="sampling", metadata={"help":"MoE second expert policy"})
    moe_top1_expert: bool = field(default=False, metadata={"help":"Use Top1 gate"})
    moe_eval_capacity_token_fraction: float = field(default=0.25, metadata={"help":"MoE capacity fraction"})

    share_decoder_input_output_embed: bool = field(default=False, metadata={"help": "share decoder input and output embeddings"})
    decoder_layers: int = field(default=32, metadata={"help": "number of decoder layers"})
    decoder_embed_dim: int = field(default=4096, metadata={"help": "decoder embedding dimension"})
    decoder_ffn_embed_dim: int = field(default=14336, metadata={"help": "decoder ffn embed dimension"})
    decoder_attention_heads: int = field(default=64, metadata={"help": "decoder attention heads"})
    moe_freq: int = field(default=1, metadata={"help": "frequency of MoE layers"})
    moe_gating_use_fp32: bool = field(default=False, metadata={"help": "Use FP32 computations in MoE gating"})
    moe_normalize_expert_grad: str = field(default="sqrt_world_size", metadata={"help":"normalize expert grad method"})

    dropout: float = field(default=0.1, metadata={"help": "dropout probability"})
    attention_dropout: float = field(default=0.1, metadata={"help": "attention dropout probability"})
    checkpoint_activations: bool = field(default=False, metadata={"help":"whether to use gradient checkpointing"})
    max_position_embeddings: int = field(default=1024, metadata={"help": "Max position embeddings, same as MixtralConfig"})



@register_model("mixtral_lm", dataclass=MixtralLanguageModelingConfig)
class MixtralLMModel(FairseqLanguageModel):

    @classmethod
    def build_model(cls, args, task):
        default_config = MixtralConfig()
：
        if getattr(args, "decoder_embed_dim", None) is not None:
            default_config.hidden_size = args.decoder_embed_dim
        else:
            args.decoder_embed_dim = default_config.hidden_size

        if getattr(args, "decoder_ffn_embed_dim", None) is not None:
            default_config.intermediate_size = args.decoder_ffn_embed_dim
        else:
            args.decoder_ffn_embed_dim = default_config.intermediate_size

        if getattr(args, "decoder_layers", None) is not None:
            default_config.num_hidden_layers = args.decoder_layers
        else:
            args.decoder_layers = default_config.num_hidden_layers

        if getattr(args, "decoder_attention_heads", None) is not None:
            default_config.num_attention_heads = args.decoder_attention_heads
        else:
            args.decoder_attention_heads = default_config.num_attention_heads

        if getattr(args, "attention_dropout", None) is not None:
            default_config.attention_dropout = args.attention_dropout
        else:
            args.attention_dropout = default_config.attention_dropout

        if args.moe_expert_count != -1:
            default_config.num_local_experts = args.moe_expert_count
        else:
            args.moe_expert_count = default_config.num_local_experts

        if args.moe_second_expert_policy is not None:
            pass
        else:
            if default_config.num_experts_per_tok > 1:
                args.moe_second_expert_policy = "sampling"
            else:
                args.moe_second_expert_policy = "none"

        if args.moe_top1_expert is not None:
            pass
        else:
            args.moe_top1_expert = False

        if args.moe_eval_capacity_token_fraction is not None:
            pass
        else:
            args.moe_eval_capacity_token_fraction = 0.25
        from .mixtral import build_mixtral_decoder
        decoder = build_mixtral_decoder(args, task)
        return cls(decoder)

    def forward(self, src_tokens, **kwargs):
        return self.decoder(src_tokens, **kwargs)


@register_model_architecture("mixtral_lm", "mixtral_lm_arch")
def mixtral_lm_arch(args):

    args.decoder_layers = getattr(args, "decoder_layers", 32)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 4096)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 14336)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 32)
    args.dropout = getattr(args, "dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.moe_expert_count = getattr(args, "moe_expert_count", 8)
    args.moe_top1_expert = getattr(args, "moe_top1_expert", False)
    args.moe_eval_capacity_token_fraction = getattr(args, "moe_eval_capacity_token_fraction", 0.25)

