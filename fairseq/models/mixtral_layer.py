# mixtral_layer.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules.moe import MOELayer, Top1Gate, Top2Gate
from fairseq.modules import MultiheadAttention
from fairseq import utils
import logging
logger = logging.getLogger(__name__)

from transformers import MixtralConfig
from transformers.activations import ACT2FN
class MixtralRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

class MixtralBlockSparseTop2MLP(nn.Module):
    def __init__(self, embed_dim, ffn_dim, activation_fn=F.gelu):
        super().__init__()
        self.ffn_dim = ffn_dim
        self.hidden_dim = embed_dim
        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.act_fn = activation_fn

    def forward(self, hidden_states):
        return self.w2(self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states))

def make_experts(args, embed_dim, ffn_dim):
    world_size = 1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size()
    expert_list = []
    ddp_rank = utils.item(torch.distributed.get_rank()) if torch.distributed.is_initialized() else 0
    start_seed = torch.randint(1000000, (1,)).item()

    if args.moe_expert_count >= world_size:
        local_moe_expert_count = args.moe_expert_count // world_size
        for i in range(local_moe_expert_count):
            with utils.set_torch_seed(start_seed + ddp_rank * local_moe_expert_count + i):
                expert_list.append(MixtralBlockSparseTop2MLP(embed_dim, ffn_dim))
    else:
        # 情况：expert少于world size
        with utils.set_torch_seed(start_seed + ddp_rank % args.moe_expert_count):
            expert_list.append(MixtralBlockSparseTop2MLP(embed_dim, ffn_dim))
    experts = nn.ModuleList(expert_list)
    return experts

class MixtralBlockSparseTop2MLP(nn.Module):
    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class MixtralSparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.Qmode = config.Qmode
        self.NumHotExperts = config.NumHotExperts
        self.Qpre = config.Qpre
        self.Q = None
        Q_F = {
            'int8':self.quantize_dequantize_bfloat16,
            'int4':self.quantize_dequantize_bfloat16_int4,
            'fp8':self.quantize_dequantize_fp8,
            'fp8base':self.quantize_dequantize_fp8_baseline
        }
        self.Q = Q_F[self.Qpre]
        
        logger.info(f"Quantizatin mode {self.Qmode}")

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = nn.ModuleList([MixtralBlockSparseTop2MLP(config) for _ in range(self.num_experts)])
        
        # Jitter parameters
        self.jitter_noise = config.router_jitter_noise

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        # act_quant = QuantIdentity(act_quant=Int8ActPerTensorFloat,return_quant_tensor=True)
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        unq_selected_experts, counts = torch.unique(selected_experts, return_counts=True)
        # hot_experts = unq_selected_experts[counts.argmax()]
        num_hot = min(self.NumHotExperts, len(unq_selected_experts))
        top_hot_experts = unq_selected_experts[torch.topk(counts, num_hot).indices]
        # logger.info(f"layer: {id(self)} selected_experts: {selected_experts.reshape(-1).tolist()}")

        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        # hidden_states = self.quantize_dequantize_bfloat16(hidden_states)
        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            if expert_idx in top_hot_experts and self.Qmode == 1 or self.Qmode==2:
                current_state = self.Q(current_state,before_expert=True)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            if expert_idx in top_hot_experts and self.Qmode == 1 or self.Qmode==2:
                current_hidden_states = self.Q(current_hidden_states, before_expert=False)
            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        # final_hidden_states = self.quantize_dequantize_bfloat16(final_hidden_states)
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

class MixtralDecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = nn.Dropout(args.dropout)
        self.normalize_before = getattr(args, "decoder_normalize_before", False)

        # 判断是否是MoE层的逻辑根据args
        # 这里示意为: 当moe_expert_count>0时则是MoE层 (可根据实际需要添加条件)
        self.is_moe_layer = (getattr(args, "moe_expert_count", 0) > 0 and getattr(args,"moe_freq",0)>0)

        self.self_attn = MultiheadAttention(
            self.embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True
        )
        self.self_attn_layer_norm = MixtralRMSNorm(self.embed_dim)

        if self.is_moe_layer:
            # MoE相关
            if args.moe_top1_expert:
                gate = Top1Gate(self.embed_dim, args.moe_expert_count, use_fp32=False,
                                moe_eval_capacity_token_fraction=args.moe_eval_capacity_token_fraction)
            else:
                gate = Top2Gate(
                    self.embed_dim,
                    args.moe_expert_count,
                    use_fp32=False,
                    second_expert_policy=getattr(args,"moe_second_expert_policy","sampling"),
                    normalize_gate_prob_before_dropping=False,
                    moe_eval_capacity_token_fraction=args.moe_eval_capacity_token_fraction,
                )
            experts = make_experts(args, self.embed_dim, args.decoder_ffn_embed_dim)
            self.moe_layer = MOELayer(gate, experts, args)
            self.final_layer_norm = MixtralRMSNorm(self.embed_dim)
        else:
            self.activation_fn = utils.get_activation_fn(activation=getattr(args, "activation_fn", "gelu"))
            self.activation_dropout_module = nn.Dropout(getattr(args, "activation_dropout", 0))
            self.fc1 = nn.Linear(self.embed_dim, args.decoder_ffn_embed_dim)
            self.fc2 = nn.Linear(args.decoder_ffn_embed_dim, self.embed_dim)
            self.final_layer_norm = MixtralRMSNorm(self.embed_dim)

    def residual_connection(self, x, residual):
        return residual + x

    def forward(self, x, incremental_state=None, **kwargs):
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=None,
            need_weights=False,
            attn_mask=None,
            incremental_state=incremental_state,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        l_aux = None
        if self.is_moe_layer:
            x = x.transpose(0,1) # B, T, C
            x, l_aux = self.moe_layer(x)
            x = x.transpose(0,1)
        else:
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)

        x = self.residual_connection(x, residual)

        if not self.normalize_before:
            x = self.final_layer_norm(x)

        return x, l_aux
