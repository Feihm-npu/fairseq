# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: This is a mirror of the code in
# https://github.com/facebookresearch/fairscale/tree/master/fairscale/nn/moe

import logging
import time
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast
import numpy as np
import os
import torch
import torch.distributed as dist
from torch import Tensor
from torch.cuda import Event as CudaEvent
from torch.nn import Module, ModuleList
from fairseq import distributed_utils

if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module

try:
    # To enable Tutel MoE optimizations:
    #   python3 -m pip install --user --upgrade git+https://github.com/microsoft/tutel@v0.1.x
    from tutel import moe as tutel_moe

    has_tutel, fused_cumsum_sub_one = True, tutel_moe.fast_cumsum_sub_one
except ModuleNotFoundError:
    has_tutel, fused_cumsum_sub_one = False, lambda mask: torch.cumsum(mask, dim=0) - 1

logger = logging.getLogger(__name__)
logger.disabled = False  

def log_cpu_affinity():
    return os.getpid()
# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.

class _AllToAll_fp(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor) -> Tensor:  # type: ignore
        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        without_q_cuda_start = torch.cuda.Event(enable_timing=True)
        without_q_cuda_end = torch.cuda.Event(enable_timing=True)
        without_q_cuda_start.record()
        if torch.distributed.is_initialized():
            dist.all_to_all_single(output, input, group=group,async_op=False)
        else:
            assert group is None
            output = input
        without_q_cuda_end.record()
        torch.cuda.synchronize()
        print(f'{log_cpu_affinity()} New cuda time without quantization in fp:',without_q_cuda_start.elapsed_time(without_q_cuda_end))
        # end = time.time()
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, _AllToAll_fp.apply(ctx.group, *grad_output))


# Based on https://github.com/pytorch/pytorch/pull/40762
class _AllToAll_int8(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor) -> Tensor:  # type: ignore
        ctx.group = group
        # Save the original dtype and shape for backward conversion
        ctx.original_dtype = input.dtype
        ctx.original_shape = input.shape
        # if ctx.convert_time == None:
        #     ctx.convert_time = 0
        # Ensure input is contiguous
        input = input.contiguous()
        # Convert to float if the input is half precision
        if input.dtype == torch.float16:
            # start_time = time.time()
            input = input.to(torch.float32)
            # end_time = time.time()
            # ctx.convert_time +=end_time-start_time
            # logger.info(f'convert time {end_time-start_time}')
        # Define scale and zero_point
        scale = 0.1
        zero_point = 0
        
        # Convert to qint8
        input_qint8 = torch.quantize_per_tensor(input, scale, zero_point, torch.qint8)
        
        # Perform the all_to_all operation
        output_qint8 = torch.empty_like(input_qint8.int_repr(), dtype=torch.int8)
        # start = time.time()
        without_q_cuda_start = torch.cuda.Event(enable_timing=True)
        without_q_cuda_end = torch.cuda.Event(enable_timing=True)
        without_q_cuda_start.record()
        if torch.distributed.is_initialized():
            dist.all_to_all_single(output_qint8, input_qint8.int_repr(), group=group,async_op=False)
        else:
            assert group is None
            output_qint8 = input_qint8.int_repr()
        without_q_cuda_end.record()
        torch.cuda.synchronize()
        print(f'{log_cpu_affinity()} New cuda time without quantization in int:',without_q_cuda_start.elapsed_time(without_q_cuda_end))
        # print(input_qint8.shape,input_qint8.dtype)
        # end = time.time()
        # Create a quantized tensor from the received int8 values
        output_qint8 = torch._make_per_tensor_quantized_tensor(output_qint8, scale, zero_point)
        
        # Convert back to original dtype and ensure it is contiguous
        output_qint8 = output_qint8.dequantize().contiguous()
        # Convert back to the original dtype
        if ctx.original_dtype == torch.float16:
            output_qint8 = output_qint8.to(torch.float16)
        return output_qint8

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, _AllToAll_int8.apply(ctx.group, *grad_output))

class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Any) -> Tensor:  # type: ignore
        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        if torch.distributed.is_initialized():
            dist.all_to_all_single(output, input, group=group, async_op=False)
        else:
            assert group is None
            output = input
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, _AllToAll.apply(ctx.group, *grad_output))

class _AllToAll_async(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Any) -> Tensor:  # type: ignore
        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        if torch.distributed.is_initialized():
            dist.all_to_all_single(output, input, group=group, async_op=True)
        else:
            assert group is None
            output = input
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, _AllToAll_async.apply(ctx.group, *grad_output))

# Replace with actual log_cpu_affinity implementation
def log_cpu_affinity():
    return "CPU Affinity log"


# class MOELayer(Base):
#     """MOELayer module which implements MixtureOfExperts as described in Gshard_.
#     ::

#         gate = Top2Gate(model_dim, num_experts)
#         moe = MOELayer(gate, expert)
#         output = moe(input)
#         l_aux = moe.l_aux

#     .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

#     Args:
#         gate (torch.nn.Module):
#             gate network
#         expert (torch.nn.Module):
#             expert network
#     """

#     def __init__(self, gate: Module, experts: Union[Module, ModuleList], args, group: Optional[Any] = None, all2all_group: Optional[Any] = None) -> None:
#         super().__init__()
#         self.gate = gate
#         if type(experts) == ModuleList:
#             self.experts = cast(ModuleList, experts)
#         else:
#             self.experts = ModuleList([experts])
#         self.expert_group = group if group is not None else distributed_utils.get_moe_group(args.moe_expert_count)
#         self.all2all_group = all2all_group if all2all_group is not None else distributed_utils.get_all2all_group(args.moe_expert_count)
#         for p in experts.parameters():
#             p.expert = True  # type: ignore
#         self.world_size = distributed_utils.get_world_size(self.expert_group)
#         self.all2all_size = distributed_utils.get_world_size(self.all2all_group)
#         self.num_local_experts = len(self.experts)
#         self.args = args
#         self.in_generation = False
#         self.a2a_cuda_event_intervals = []
#         self.a2a_cpu_time_ms = 0.0

#     def forward(self, *input: Tensor, input_padding_mask=None, **kwargs: Any) -> Tensor:
#         assert len(input) == 1, "only single input Tensor supported"
#         input = input[0]
#         assert len(input.shape) == 3, "input Tensor must have dimensions: (s)equence, (t)oken, (m)odel"
#         if input_padding_mask is not None:
#             assert len(input_padding_mask.shape) == 2, "input Tensor must have dimensions: (s)equence, (t)oken"
#             assert input_padding_mask.shape[0] == input.shape[0]
#             assert input_padding_mask.shape[1] == input.shape[1]
#         # assert input.shape[0] % len(self.experts) == 0, "num tokens must be order of number of local experts"

#         # Implement Algorithm 2 from GShard paper.
#         d_model = input.shape[2]
#         # Pad to expected batch size
#         input_shape = list(input.shape)
#         expected_bsz = getattr(self.args, 'batch_size', 0) if self.training else getattr(self.args, 'batch_size_valid', 0)
#         # This indicates that --batch-size or --max-sentences is not specified
#         if expected_bsz is None:
#             expected_bsz = 0
#         # Note: Padding is not necessary at generation time at present
#         # because all DDP workers process the same batch. Also, batch size at generation time
#         # can be different from that present in the checkpoint state
#         if not self.in_generation and expected_bsz != 0 and input_shape[0] != expected_bsz:
#             logger.warning(f"padding batch with unexpected size {input_shape[0]} (expected: {expected_bsz})")
#             assert input_shape[0] < expected_bsz, f"{input_shape[0]} < {expected_bsz}"
#             padded_input = torch.zeros(
#                 (expected_bsz, input_shape[1], input_shape[2]),
#                 dtype=input.dtype, layout=input.layout, device=input.device)
#             padded_input[:input_shape[0], :, :] = input
#             input = padded_input

#             padded_input_padding_mask = torch.ones(
#                 (expected_bsz, input_shape[1], ), dtype=torch.bool, device=input.device
#             )
#             if input_padding_mask is not None:
#                 padded_input_padding_mask[:input_shape[0], :] = input_padding_mask
#             else:
#                 padded_input_padding_mask[:input_shape[0], :] = False
#             input_padding_mask = padded_input_padding_mask

#         # Reshape into S tokens by dropping sequence dimension.
#         reshaped_input = input.reshape(-1, d_model)
#         reshaped_input_shape = reshaped_input.shape
#         reshaped_input_padding_mask = input_padding_mask.reshape(-1) if input_padding_mask is not None else None

#         # Doing padding here when --max-tokens is specified and not --batch-size or --max-sentences
#         # Pro of --max-tokens: more flexible for MT variable sequence lengths
#         # Con of --max-tokens: extra all-reduce needed to figure out optimal padding without running OOM
#         if expected_bsz == 0:
#             expected_dim = int(distributed_utils.all_reduce(
#                 reshaped_input_shape[0] * torch.ones((1,), dtype=torch.long, device=input.device),
#                 group=dist.group.WORLD,
#                 op="max",
#             ).item())
#             padded_input = torch.zeros(
#                 (expected_dim, reshaped_input_shape[1]),
#                 dtype=input.dtype, layout=input.layout, device=input.device)
#             padded_input[:reshaped_input_shape[0], :] = reshaped_input
#             reshaped_input = padded_input

#             padded_input_padding_mask = torch.ones(
#                 (expected_dim,), dtype=torch.bool, device=padded_input.device
#             )
#             if reshaped_input_padding_mask is not None:
#                 padded_input_padding_mask[:reshaped_input_shape[0]] = reshaped_input_padding_mask
#             else:
#                 padded_input_padding_mask[:reshaped_input_shape[0]] = False
#             reshaped_input_padding_mask = padded_input_padding_mask

#         if has_tutel:
#             l_aux, self.metadata, C, E, indices_, locations_, gates_ = self.gate(reshaped_input, reshaped_input_padding_mask)
#             S, M = reshaped_input.size(0), reshaped_input.size(1)

#             if not hasattr(self, '_tutel_dispatcher'):
#                 self._tutel_dispatcher = tutel_moe.fast_dispatcher(E, C, M, dispatch_dtype=reshaped_input.dtype)
#             self._tutel_dispatcher.update(indices_, locations_, gates_, capacity=C)
#             dispatched_input = self._tutel_dispatcher.encode(reshaped_input)
#         else:
#             l_aux, combine_weights, dispatch_mask, self.metadata = self.gate(reshaped_input, reshaped_input_padding_mask)

#             dispatch_mask = dispatch_mask.to(input.dtype).permute(1, 2, 0)  # S,E,C -> E,C,S
#             E, C, S = dispatch_mask.size()
#             M = reshaped_input.size(1)
#             assert reshaped_input.size() == (S, M)
#             # einsum("sec,sm->ecm")
#             dispatched_input = torch.mm(dispatch_mask.view(E*C, S), reshaped_input)  # -> (E*C),M

#         if self.all2all_size > 1:
#             dispatched_input = self.all_to_all_wrapper(dispatched_input)

#         # Re-shape after all-to-all: ecm -> gecm
#         dispatched_input = dispatched_input.reshape(self.all2all_size, self.num_local_experts, -1, d_model)
#         chunks = dispatched_input.chunk(self.num_local_experts, dim=1)
#         expert_outputs = []
#         for chunk, expert in zip(chunks, self.experts):
#             expert_outputs += [expert(chunk)]
#         expert_output = torch.cat(expert_outputs, dim=1)

#         if self.all2all_size > 1:
#             expert_output = self.all_to_all_wrapper(expert_output)

#         # Re-shape back: gecm -> ecm
#         expert_output = expert_output.reshape(self.all2all_size * self.num_local_experts, -1, d_model)

#         if has_tutel:
#             combined_output = self._tutel_dispatcher.decode(expert_output.view(E*C, M))
#         else:
#             # einsum("sec,ecm->sm")
#             combined_output = combine_weights.view(S, E*C).mm(expert_output.view(E*C, M))

#         # Remove padding here when --max-tokens is specified and not --batch-size or --max-sentences
#         combined_output = combined_output[:reshaped_input_shape[0], :]
#         combined_output = combined_output.reshape(input.shape)
#         combined_output = combined_output[:input_shape[0], :, :]

#         self.record_all_to_all_stats()

#         return combined_output, l_aux

#     def prepare_for_inference_(self):
#         self.in_generation = True

#     def all_to_all_wrapper(self, input: Tensor):
#         dummy_a2a = getattr(self.args, 'dummy_a2a', False)
#         if dummy_a2a:
#             input = input.contiguous()
#             output = input.detach().clone()
#             return input
#         # always record times, since it is not a lot of overhead
#         # if we do not log it we simply clear it off in record_all_to_all_stats
#         cuda_start = torch.cuda.Event(enable_timing=True)
#         cuda_end = torch.cuda.Event(enable_timing=True)
#         cpu_start = time.time() * 1000
#         cuda_start.record()
#         output = _AllToAll.apply(self.all2all_group, input)
#         cuda_end.record()
#         cpu_end = time.time() * 1000
#         self.a2a_cpu_time_ms += (cpu_end - cpu_start)
#         self.a2a_cuda_event_intervals.append((cuda_start, cuda_end))
#         return output

#     def record_all_to_all_stats(self):
#         # controlled via an argument as we want to minimize any impact from torch.cuda.synchronize()
#         record_a2a_perf_stats = getattr(self.args, 'record_a2a_perf_stats', False)
#         if record_a2a_perf_stats:
#             torch.cuda.synchronize()
#             self.metadata["all_to_all_cpu_time_ms"] = self.a2a_cpu_time_ms
#             a2a_cuda_time_ms = 0.0
#             for ev_start, ev_end in self.a2a_cuda_event_intervals:
#                 a2a_cuda_time_ms += ev_start.elapsed_time(ev_end)
#             self.metadata["all_to_all_cuda_time_ms"] = a2a_cuda_time_ms
#         # reset stats
#         self.a2a_cpu_time_ms = 0.0
#         self.a2a_cuda_event_intervals = []


class MOELayer(Base):
    """MOELayer module which implements MixtureOfExperts as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)
        moe = MOELayer(gate, expert)
        output = moe(input)
        l_aux = moe.l_aux

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        gate (torch.nn.Module):
            gate network
        expert (torch.nn.Module):
            expert network
    """

    def __init__(self, gate: Module, experts: Union[Module, ModuleList], args, group: Optional[Any] = None, all2all_group: Optional[Any] = None) -> None:
        super().__init__()
        self.gate = gate
        if type(experts) == ModuleList:
            self.experts = cast(ModuleList, experts)
        else:
            self.experts = ModuleList([experts])
        self.expert_group = group if group is not None else distributed_utils.get_moe_group(args.moe_expert_count)
        self.all2all_group = all2all_group if all2all_group is not None else distributed_utils.get_all2all_group(args.moe_expert_count)
        for p in experts.parameters():
            p.expert = True  # type: ignore
        self.world_size = distributed_utils.get_world_size(self.expert_group)
        self.all2all_size = distributed_utils.get_world_size(self.all2all_group)
        self.num_local_experts = len(self.experts)
        self.args = args
        self.in_generation = False
        self.a2a_cuda_event_intervals = []
        self.a2a_cpu_time_ms = 0.0
        self.total_cpu = 0.0
        self.total_gpu = 0.0
        self.total_forward = 0
        # Correctly get the rank
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.global_rank2 = dist.get_rank(self.all2all_group)
            self.global_rank3 = dist.get_rank(self.expert_group)
            self.global_rank4 = dist.get_global_rank(self.expert_group,0)
            self.global_rank5 = dist.get_global_rank(self.all2all_group,0)
        else:
            self.rank = 0
            self.global_rank2 = 0
            self.global_rank3 = 0
        # q_mode all2all mode
        # 0: fp16/fp32
        # 1: all int8
        # 2: int8 for only hot experts
        self.q_mode = getattr(self.args, 'Qmode', 0)
        logger.info(f'All2All communication mode: {self.q_mode}')
        if self.q_mode!=0:
            self.gate.q_mode = True
        self.gate.q_mode = False
        logger.info(f'Gate q_mode: {self.gate.q_mode}')
        
        # logger.info(f'Rank: {self.rank},{self.global_rank2},{self.global_rank3},{self.global_rank4},{self.global_rank5}')
        # logger.info(f'world_size {self.world_size}, all2all_size {self.all2all_size}')

    def forward(self, *input: Tensor, input_padding_mask=None, **kwargs: Any) -> Tensor:
        self.rank = dist.get_rank()
        # logger.info(f'Inside forward rank {self.rank}')
        self.total_forward += 1
        assert len(input) == 1, "only single input Tensor supported"
        input = input[0]
        assert len(input.shape) == 3, "input Tensor must have dimensions: (s)equence, (t)oken, (m)odel"
        if input_padding_mask is not None:
            assert len(input_padding_mask.shape) == 2, "input Tensor must have dimensions: (s)equence, (t)oken"
            assert input_padding_mask.shape[0] == input.shape[0]
            assert input_padding_mask.shape[1] == input.shape[1]

        d_model = input.shape[2]
        input_shape = list(input.shape)
        expected_bsz = getattr(self.args, 'batch_size', 0) if self.training else getattr(self.args, 'batch_size_valid', 0)
        if expected_bsz is None:
            expected_bsz = 0
        if not self.in_generation and expected_bsz != 0 and input_shape[0] != expected_bsz:
            logger.warning(f"padding batch with unexpected size {input_shape[0]} (expected: {expected_bsz})")
            assert input_shape[0] < expected_bsz, f"{input_shape[0]} < {expected_bsz}"
            padded_input = torch.zeros(
                (expected_bsz, input_shape[1], input_shape[2]),
                dtype=input.dtype, layout=input.layout, device=input.device)
            padded_input[:input_shape[0], :, :] = input
            input = padded_input

            padded_input_padding_mask = torch.ones(
                (expected_bsz, input_shape[1],), dtype=torch.bool, device=input.device
            )
            if input_padding_mask is not None:
                padded_input_padding_mask[:input_shape[0], :] = input_padding_mask
            else:
                padded_input_padding_mask[:input_shape[0], :] = False
            input_padding_mask = padded_input_padding_mask

        reshaped_input = input.reshape(-1, d_model)
        reshaped_input_shape = reshaped_input.shape
        reshaped_input_padding_mask = input_padding_mask.reshape(-1) if input_padding_mask is not None else None

        if has_tutel:
            l_aux, self.metadata, C, E, indices_, locations_, gates_ = self.gate(reshaped_input, reshaped_input_padding_mask)
            # logger.info(f'capactity: {C}, experts: {E}')
            S, M = reshaped_input.size(0), reshaped_input.size(1)

            if not hasattr(self, '_tutel_dispatcher'):
                self._tutel_dispatcher = tutel_moe.fast_dispatcher(E, C, M, dispatch_dtype=reshaped_input.dtype)
            self._tutel_dispatcher.update(indices_, locations_, gates_, capacity=C)
            dispatched_input = self._tutel_dispatcher.encode(reshaped_input)
        else:
            l_aux, combine_weights, dispatch_mask, self.metadata = self.gate(reshaped_input, reshaped_input_padding_mask)

            dispatch_mask = dispatch_mask.to(input.dtype).permute(1, 2, 0)  # S,E,C -> E,C,S
            E, C, S = dispatch_mask.size()
            M = reshaped_input.size(1)
            assert reshaped_input.size() == (S, M)
            dispatched_input = torch.mm(dispatch_mask.view(E * C, S), reshaped_input)  # -> (E*C),M

        self.a2a_cpu_time_ms = 0
        self.a2a_cuda_event_intervals = []

        

        overflow_expert1 = self.metadata["overflow_expert1"]
        overflow_expert2 = self.metadata["overflow_expert2"]
        # logger.info("overflow_expert1", overflow_expert1)
        hot_experts = (overflow_expert1 > 0) | (overflow_expert2 > 0)
        # to find the local ones
        local_expert_start_idx = self.rank * self.num_local_experts
        local_expert_end_idx = local_expert_start_idx + self.num_local_experts
        local_expert_indices = list(range(local_expert_start_idx, local_expert_end_idx))
        # Get hot_experts for local experts
        local_hot_experts = hot_experts[local_expert_indices]
        # logger.info(f'{dist.get_rank()} hot_experts:{local_hot_experts}')
        num_experts = hot_experts.shape[0]
        # print("HOT Experts:",hot_experts)
        dispatched_input = dispatched_input.reshape(self.all2all_size, self.num_local_experts, -1, d_model)

        if self.q_mode==1:
            q_begin = time.time()
            dispatched_input,min_val,scale = self.quantize_int8(dispatched_input)
            q_end = time.time()
            logger.info(f'quantize_int8 time before expert: {q_end-q_begin}')

        if self.all2all_size > 1:
            start_time = time.time()
            # Combine tensors before all_to_all if q_mode == 1
            if self.q_mode == 1:
                # Assuming dispatched_input, min_val, and scale are compatible for concatenation along the last dimension
                # min_val = self.all_to_all_wrapper(min_val)
                # scale = self.all_to_all_wrapper(scale)
                # dispatched_input = self.all_to_all_wrapper(dispatched_input, quantized=True)

                dispatched_input,scale, min_val, gpu_time = self.all_to_all_wrapper(dispatched_input, quantized=True, scale=scale, min_val=min_val)
                # combined_input = torch.cat([dispatched_input, min_val, scale], dim=-2)
                # combined_output = self.all_to_all_wrapper(combined_input)

                # # Split the combined output back into dispatched_input, min_val, and scale
                # dispatched_input, min_val, scale = torch.split(combined_output, [dispatched_input.size(-2), min_val.size(-2), scale.size(-2)], dim=-2)
            else:
                dispatched_input, gpu_time = self.all_to_all_wrapper(dispatched_input)

            end_time = time.time()
            logger.info(f'All to all time before expert: {end_time - start_time}, GPU time: {gpu_time}')
        if self.q_mode==1:
            dq_time = time.time()
            dispatched_input = self.dequantize_int8(dispatched_input,min_val,scale)
            dq_end_time = time.time()
            logger.info(f'Dequantize time before expert: {dq_end_time-dq_time}')

        ## Expert calculation
        all_expert_outputs = []
        start_time = time.time()
        for expert_index, (chunk, expert) in enumerate(zip(dispatched_input.chunk(self.num_local_experts, dim=1), self.experts)):
        #     # print("Expert index",expert_index)
        #     if self.q_mode==1 or (local_hot_experts[expert_index] and self.q_mode==2):
        #     # if True:
        #         logger.info(f'{dist.get_rank()} find one hot expert')
        #         start_time = time.time()
        #         chunk = self.quantize_dequantize_fp16(chunk)
        #         end_time = time.time()
        #         logger.info(f'quantize_dequantize_fp16 time for input: {end_time-start_time}')
        #         # chunk = self.all_to_all_wrapper(chunk, quantized=True)
        #     else:
        #         logger.info(f'{self.rank} find one normal expert')
        #         # chunk = self.all_to_all_wrapper(chunk, quantized=True)
            
            all_expert_outputs.append(expert(chunk))
        end_time = time.time()
        logger.info(f'expert calculating time: {end_time-start_time}')
  
        ## Quantize and send the expert outputs 
        expert_outputs2 = []
        min_vals,scales = [],[]
        if self.q_mode==1 or self.q_mode==2:
            start_time = time.time()
            for idx, output in enumerate(all_expert_outputs):
                if self.q_mode==1 or local_hot_experts[idx]:                     
                    output,min_val,scale = self.quantize_int8(output)
                expert_outputs2.append(output)
                min_vals.append(min_val)
                # logger.info(f'min_val: {min_val}, min_val shape {min_val.shape}')
                scales.append(scale)
            end_time = time.time()
            logger.info(f'quantize_int8 time after expert: {end_time-start_time}')
            # Start all to all communication
            expert_output = torch.cat(expert_outputs2, dim=1)
            min_val_list = torch.cat(min_vals,dim=1)
            scale_list = torch.cat(scales,dim=1)
            
            if self.all2all_size > 1:
                start_time = time.time()
                # min_val_list = self.all_to_all_wrapper(min_val_list, quantized=False)
                # scale_list = self.all_to_all_wrapper(scale_list, quantized=False)
                # expert_output = self.all_to_all_wrapper(expert_output, quantized=True)
                expert_output,scale_list, min_val_list, gpu_time = self.all_to_all_wrapper(expert_output, quantized=True, scale=scale_list, min_val=min_val_list)
                # combined_input = torch.cat([expert_output, min_val_list, scale_list], dim=-2)
                # combined_input = self.all_to_all_wrapper(combined_input, quantized=False)
                # expert_output, min_val_list, scale_list = torch.split(combined_input, [expert_output.size(-2), min_val_list.size(-2), scale_list.size(-2)], dim=-2)
                end_time = time.time()
                logger.info(f'All to all time after expert: {end_time-start_time}, GPU time: {gpu_time}')
        elif self.q_mode==0:
            expert_output = torch.cat(all_expert_outputs, dim=1)
            if self.all2all_size > 1:
                start_time = time.time()
                expert_output, gpu_time = self.all_to_all_wrapper(expert_output, quantized=False)
                end_time = time.time()
                logger.info(f'All to all time after expert: {end_time-start_time}, GPU time: {gpu_time}')
        ## Dequantize the expert outputs if needed  
        expert_outputs_q = []
        if self.q_mode==1 or self.q_mode==2:
            dq_time = time.time()
            for idx, (output,min,s) in enumerate(zip(expert_output.chunk(self.num_local_experts,dim=1),min_val_list.chunk(self.num_local_experts,dim=1),scale_list.chunk(self.num_local_experts,dim=1))):
                if self.q_mode==1 or local_hot_experts[idx]:
                    expert_output_q = self.dequantize_int8(output,min,s)
                expert_outputs_q.append(expert_output_q)
            dq_end_time = time.time()
            logger.info(f'Dequantize time after expert: {dq_end_time-dq_time}')
            expert_outputs_q = torch.cat(expert_outputs_q, dim=1)
        else:
            expert_outputs_q = expert_output
        
        expert_output = expert_outputs_q.reshape(self.all2all_size * self.num_local_experts, -1, d_model)

        if has_tutel:
            combined_output = self._tutel_dispatcher.decode(expert_output.view(E * C, M))
        else:
            combined_output = combine_weights.view(S, E * C).mm(expert_output.view(E * C, M))

        combined_output = combined_output[:reshaped_input_shape[0], :]
        combined_output = combined_output.reshape(input.shape)
        combined_output = combined_output[:input_shape[0], :, :]
        # self.record_all_to_all_stats()
        return combined_output, l_aux

    def combine_tensors(self, input):
        assert input.dtype == torch.int8, "Input tensor must be of type int8"
        # Assuming the last dimension is the one to pack the int8 values
        # Combine int8 into fp16
        input_shape = input.shape  # Save the original shape [16, 32, 698, 768]
        assert input_shape[-1] % 2 == 0, "The last dimension size must be even to combine int8 into float16"
        
        # Reshape and combine two consecutive int8 values
        input_int8 = input.view(*input_shape[:-1], input_shape[-1] // 2, 2).type(torch.uint8).to(torch.int16)
        input_combined = (input_int8[..., 0] << 8) | (input_int8[..., 1])
        return input_combined, input_shape

    def decombine_tensors(self, input_combined, input_shape):
        assert input.dtype == torch.float16, "Input tensor must be of type float16"
        input_combined = input_combined.view(torch.int16)
        input_first = (output >> 8).type(torch.uint8).view(torch.int8)  # First int8 from higher 8 bits
        input_second = (output & 0xFF).type(torch.uint8).view(torch.int8) # Second int8 from lower 8 bits
        
        # Combine them back into original shape
        output = torch.stack([input_first, input_second], axis=-1).reshape(*input_shape)
        return output

    def quantize_int8(self, tensor_fp16):
        # Ensure the tensor is of type fp16
        assert tensor_fp16.dtype == torch.float16, "Input tensor must be of type fp16"
        # Step 1: Normalize the fp16 values to fit into the range of int8
        # logger.info(f'Inside quantize int8 function, input shape: {tensor_fp16.shape}')
        # min_val = tensor_fp16.min()
        # max_val = tensor_fp16.max()
        max_val = tensor_fp16.max(dim=2,keepdim=True)[0]
        min_val = tensor_fp16.min(dim=2,keepdim=True)[0]
        # Avoid division by zero if min and max are the same
        scale = max_val - min_val
        scale[scale == 0] = 1.0
        normalized_fp16 = (tensor_fp16 - min_val) / scale
        # Step 2: Quantize the normalized values into int8
        tensor_qint8 = (normalized_fp16 * 255 - 128).round().clamp(-128, 127).to(torch.int8)

        return tensor_qint8,min_val,scale

    def dequantize_int8(self, tensor_int8,min_val,scale):
        assert tensor_int8.dtype == torch.int8, "Input tensor must be of type int8"
        dequantized_fp16 = (tensor_int8.float() + 128) / 255 * scale + min_val
        return dequantized_fp16.half()


    def prepare_for_inference_(self):
        self.in_generation = True

    def all_to_all_wrapper(self, input: torch.Tensor, quantized=False, scale=None, min_val=None):
        dummy_a2a = getattr(self.args, 'dummy_a2a', False)
        if dummy_a2a:
            input = input.contiguous()
            output = input.detach().clone()
            return input

        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        cpu_start = time.time() * 1000
        cuda_start.record()
        quantized = False
        if scale is not None:
            scale = _AllToAll.apply(self.all2all_group, scale)
            min_val = _AllToAll.apply(self.all2all_group, min_val)

        if quantized:
            assert input.dtype == torch.int8, "Input tensor must be of type int8"
            # Assuming the last dimension is the one to pack the int8 values
            # Combine int8 into fp16
            input_shape = input.shape  # Save the original shape [16, 32, 698, 768]
            assert input_shape[-1] % 2 == 0, "The last dimension size must be even to combine int8 into float16"
            
            # Reshape and combine two consecutive int8 values
            input_int8 = input.view(*input_shape[:-1], input_shape[-1] // 2, 2).type(torch.uint8).to(torch.int16)
            input = (input_int8[..., 0] << 8) | (input_int8[..., 1])

        # Send the combined input through _AllToAll.apply
        # output = _AllToAll.apply(self.all2all_group, input.view(torch.float16))
        
            output = _AllToAll.apply(self.all2all_group, input.view(torch.float16))
        else:
            output = _AllToAll.apply(self.all2all_group, input)

        # Decompose float16 back to int8 after receiving the output
        if quantized:
            output = output.view(torch.int16)
            output_int8_first = (output >> 8).type(torch.uint8).view(torch.int8)  # First int8 from higher 8 bits
            output_int8_second = (output & 0xFF).type(torch.uint8).view(torch.int8) # Second int8 from lower 8 bits
            
            # Combine them back into original shape
            output = torch.stack([output_int8_first, output_int8_second], axis=-1).reshape(*input_shape)

        
        cuda_end.record()
        torch.cuda.synchronize()
        # print("New cuda time:", cuda_start.elapsed_time(cuda_end))
        cpu_end = time.time() * 1000
        self.a2a_cpu_time_ms += (cpu_end - cpu_start)
        self.a2a_cuda_event_intervals.append((cuda_start, cuda_end))
        if scale is not None:
            return output, scale, min_val, cuda_start.elapsed_time(cuda_end)
        return output, cuda_start.elapsed_time(cuda_end)

    def record_all_to_all_stats(self):
        record_a2a_perf_stats = True
        if record_a2a_perf_stats:
            torch.cuda.synchronize()
            self.total_cpu += self.a2a_cpu_time_ms

            a2a_cuda_time_ms = 0.0
            for ev_start, ev_end in self.a2a_cuda_event_intervals:
                a2a_cuda_time_ms += ev_start.elapsed_time(ev_end)
            self.total_gpu += a2a_cuda_time_ms
            self.a2a_cpu_time_ms = 0.0
            self.a2a_cuda_event_intervals = []

            logger.info(
                f'cpu time: {self.total_cpu}, gpu time: {self.total_gpu}'
            )
            logger.info(
            'Expert1 overflow: {}\n'
            'Expert2 overflow: {}\n'
            'Expert1 unused count: {}\n'
            'Expert2 unused count: {}\n'
            'Expert1 balance top: {}\n'
            'Expert2 balance top: {}\n'.format(
                self.metadata["overflow_expert1"],
                self.metadata["overflow_expert2"],
                self.metadata["unused_expert1_count"],
                self.metadata["unused_expert2_count"],
                self.metadata["expert1_balance_top"],
                self.metadata["expert2_balance_top"]
            )
            )

