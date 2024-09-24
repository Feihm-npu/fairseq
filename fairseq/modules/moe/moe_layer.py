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
logger.disabled = True  

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
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor) -> Tensor:  # type: ignore
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

# Replace with actual log_cpu_affinity implementation
def log_cpu_affinity():
    return "CPU Affinity log"



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
        self.q_mode = getattr(self.args, 'Qmode', 2)
        logger.info(f'All2All communication mode: {self.q_mode}')
        if self.q_mode!=0:
            self.gate.q_mode = True
        logger.info(f'Gate q_mode: {self.gate.q_mode}')
        
        # logger.info(f'Rank: {self.rank},{self.global_rank2},{self.global_rank3},{self.global_rank4},{self.global_rank5}')
        # logger.info(f'world_size {self.world_size}, all2all_size {self.all2all_size}')

    def forward(self, *input: Tensor, input_padding_mask=None, **kwargs: Any) -> Tensor:
        self.rank = dist.get_rank()
        logger.info(f'Inside forward rank {self.rank}')
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
            logger.info(f'capactity: {C}, experts: {E}')
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

        dispatched_input = dispatched_input.reshape(self.all2all_size, self.num_local_experts, -1, d_model)

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
        expert_outputs = []
        all_expert_outputs = []
        if self.all2all_size > 1:
            dispatched_input = self.all_to_all_wrapper(dispatched_input)
        for expert_index, (chunk, expert) in enumerate(zip(dispatched_input.chunk(self.num_local_experts, dim=1), self.experts)):
            # print("Expert index",expert_index)
            if self.q_mode==1 or (local_hot_experts[expert_index] and self.q_mode==2):
            # if True:
                logger.info(f'{dist.get_rank()} find one hot expert')
                start_time = time.time()
                chunk = self.quantize_dequantize_fp16(chunk)
                end_time = time.time()
                logger.info(f'quantize_dequantize_fp16 time for input: {end_time-start_time}')
                # chunk = self.all_to_all_wrapper(chunk, quantized=True)
            else:
                logger.info(f'{self.rank} find one normal expert')
                # chunk = self.all_to_all_wrapper(chunk, quantized=True)
            start_time = time.time()
            all_expert_outputs.append(expert(chunk))
            end_time = time.time()
            logger.info(f'expert time: {end_time-start_time}')
  
        print("Number of experts",len(all_expert_outputs))
        if self.all2all_size > 1:
            expert_outputs2 = []
            min_vals,scales = []
            for idx, output in enumerate(all_expert_outputs):
                if self.q_mode==1 or (local_hot_experts[idx] and self.q_mode==2):
                # if True:
                    start_time = time.time()
                    output,min_val,scale = self.quantize_int8(output)
                    end_time = time.time()
                    logger.info(f'quantize_dequantize_fp16 time for output: {end_time-start_time}')
                # output = self.all_to_all_wrapper(output, quantized=False)
                expert_outputs2.append(output)
                min_vals.append(min_val)
                scales.append(scale)
            expert_output = torch.cat(expert_outputs2, dim=1)
            expert_output = self.all_to_all_wrapper(expert_output, quantized=False)
            min_vals = self.all_to_all_wrapper(min_vals, quantized=False)
            scales = self.all_to_all_wrapper(scales, quantized=False)
            expert_output = self.dequantize_int8(expert_out)
            # for output in expert_outputs:
            #     output = self.all_to_all_wrapper(output, quantized=False)
            #     expert_outputs2.append(output)
            # for output in hot_expert_outputs:
            #     output = self.all_to_all_wrapper(output, quantized=True)
            #     expert_outputs2.append(output)
            # expert_output = torch.cat(expert_outputs2, dim=1)
        # expert_output = torch.cat(expert_outputs, dim=1)
        # if self.all2all_size > 1:
        #     expert_output = self.all_to_all_wrapper(expert_output, is_expert=False, quantized=False)
        else:
            expert_outputs2 = []
            for idx, output in enumerate(all_expert_outputs):
                if self.q_mode==1 or (local_hot_experts[idx] and self.q_mode==2):
                # if True:
                    output = self.quantize_dequantize_fp16(output)
                expert_outputs2.append(output)
            expert_output = torch.cat(expert_outputs2, dim=1)  
            expert_output = self.all_to_all_wrapper(expert_output, quantized=False)
        expert_output = expert_output.reshape(self.all2all_size * self.num_local_experts, -1, d_model)

        if has_tutel:
            combined_output = self._tutel_dispatcher.decode(expert_output.view(E * C, M))
        else:
            combined_output = combine_weights.view(S, E * C).mm(expert_output.view(E * C, M))

        combined_output = combined_output[:reshaped_input_shape[0], :]
        combined_output = combined_output.reshape(input.shape)
        combined_output = combined_output[:input_shape[0], :, :]
        self.record_all_to_all_stats()
        return combined_output, l_aux


    def lower_precision_tensor(self, tensor: Tensor) -> Tensor:
        if tensor.dtype != torch.float16:
            tensor = tensor.to(torch.float16)
        return (tensor * 256).round() / 256

    def restore_precision_tensor(self, tensor: Tensor) -> Tensor:
        if tensor.dtype != torch.float16:
            tensor = tensor.to(torch.float16)
        return tensor / 256
    
    def quantize_dequantize(self, tensor: Tensor):
        scale = 0.1
        zero_point = 0
        # Convert to qint8
        if tensor.dtype == torch.float16:
            # start_time = time.time()
            tensor = tensor.to(torch.float32)
        input_qint8 = torch.quantize_per_tensor(tensor, scale, zero_point, torch.qint8)
        output_qint8 = torch.empty_like(input_qint8.int_repr(), dtype=torch.int8)
        output_qint8 = torch._make_per_tensor_quantized_tensor(input_qint8, scale, zero_point)
        output_qint8 = output_qint8.dequantize()
        if tensor.dtype == torch.float16:
            # start_time = time.time()
            output = output_qint8.to(torch.float16)
        return output

    
    def quantize_dequantize_fp16(self, tensor_fp16):
        # Ensure the tensor is of type fp16
        assert tensor_fp16.dtype == torch.float16, "Input tensor must be of type fp16"


        # Step 1: Normalize the fp16 values to fit into the range of int8
        min_val = tensor_fp16.min()
        max_val = tensor_fp16.max()

        # Avoid division by zero if min and max are the same
        scale = (max_val - min_val).float()
        scale[scale == 0] = 1.0

        normalized_fp16 = (tensor_fp16 - min_val) / scale

        # Step 2: Quantize the normalized values into int8
        tensor_qint8 = (normalized_fp16 * 255 - 128).round().clamp(-128, 127).to(torch.int8)

        # Step 3: Dequantize the int8 values back to fp16
        dequantized_fp16 = (tensor_qint8.float() + 128) / 255 * scale + min_val

        return dequantized_fp16.half()
    
    def quantize_int8(self, tensor_fp16):
        # Ensure the tensor is of type fp16
        assert tensor_fp16.dtype == torch.float16, "Input tensor must be of type fp16"
        # Step 1: Normalize the fp16 values to fit into the range of int8
        min_val = tensor_fp16.min()
        max_val = tensor_fp16.max()
        # Avoid division by zero if min and max are the same
        scale = (max_val - min_val).float()
        scale[scale == 0] = 1.0
        normalized_fp16 = (tensor_fp16 - min_val) / scale
        # Step 2: Quantize the normalized values into int8
        tensor_qint8 = (normalized_fp16 * 255 - 128).round().clamp(-128, 127).to(torch.int8)

        return dequantized_fp16,min_val,scale

    def dequantize_int8(self, tensor_int8,min_val,scale):
        assert tensor_fp16.dtype == torch.int8, "Input tensor must be of type int8"
        dequantized_fp16 = (tensor_int8.float() + 128) / 255 * scale + min_val
        return dequantized_fp16.half()


    def prepare_for_inference_(self):
        self.in_generation = True

    def all_to_all_wrapper(self, input: torch.Tensor, quantized=False):
        dummy_a2a = getattr(self.args, 'dummy_a2a', False)
        if dummy_a2a:
            input = input.contiguous()
            output = input.detach().clone()
            return input

        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        cpu_start = time.time() * 1000
        cuda_start.record()

        if quantized:
            # Assuming the last dimension is the one to pack the int8 values
            # Combine int8 into fp16
            input_shape = input.shape  # Save the original shape [16, 32, 698, 768]
            assert input_shape[-1] % 2 == 0, "The last dimension size must be even to combine int8 into float16"
            
            # Reshape and combine two consecutive int8 values
            input_int8 = input.view(*input_shape[:-1], input_shape[-1] // 2, 2).cpu().numpy()
            input_combined = (input_int8[..., 0].astype(np.uint16) << 8) | (input_int8[..., 1].astype(np.uint16))
            input_fp16 = input_combined.view(np.float16)
            input = torch.from_numpy(input_fp16).cuda()

        # Send the combined input through _AllToAll.apply
        output = _AllToAll.apply(self.all2all_group, input)

        # Decompose float16 back to int8 after receiving the output
        if quantized:
            output_fp16 = output.cpu().numpy().view(np.uint16)  # Interpret as uint16
            output_int8_first = (output_fp16 >> 8).astype(np.int8)  # First int8 from higher 8 bits
            output_int8_second = (output_fp16 & 0xFF).astype(np.int8)  # Second int8 from lower 8 bits
            
            # Combine them back into original shape
            output_int8 = np.stack([output_int8_first, output_int8_second], axis=-1).reshape(*input_shape)
            output = torch.from_numpy(output_int8).cuda()

        cuda_end.record()
        torch.cuda.synchronize()
        print("New cuda time:", cuda_start.elapsed_time(cuda_end))
        cpu_end = time.time() * 1000
        self.a2a_cpu_time_ms += (cpu_end - cpu_start)
        self.a2a_cuda_event_intervals.append((cuda_start, cuda_end))
        return output

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
#         self.total_cpu = 0.0
#         self.total_gpu = 0.0
#         self.total_forward = 0
#         rank = dist.get_global_rank(self.all2all_group,0) if dist.is_initialized() else 0
#         logger.info(f'Rank: {rank}')

#     def forward(self, *input: Tensor, input_padding_mask=None, **kwargs: Any) -> Tensor:
#         self.total_forward += 1
#         # print(f'now round: {self.total_forward}')
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
            
#             # if indices_[0].get_device() == 0:
#             #     ind_x, ind_y = indices_[0].detach().cpu().numpy(), indices_[1].detach().cpu().numpy()
#             #     ind_xy = np.append(ind_x, ind_y)
#             #     y_0_npy = np.load('hotness_gpu_0.npy') if os.path.isfile("hotness_gpu_0.npy") else []
#             #     np.save('hotness_gpu_0.npy', np.append(y_0_npy, ind_xy))

#             # if indices_[0].get_device() == 1:
#             #     ind_x, ind_y = indices_[0].detach().cpu().numpy(), indices_[1].detach().cpu().numpy()
#             #     ind_xy = np.append(ind_x, ind_y)
#             #     y_1_npy = np.load('hotness_gpu_1.npy') if os.path.isfile("hotness_gpu_1.npy") else []
#             #     np.save('hotness_gpu_1.npy', np.append(y_1_npy, ind_xy))

#             # if indices_[0].get_device() == 2:
#             #     ind_x, ind_y = indices_[0].detach().cpu().numpy(), indices_[1].detach().cpu().numpy()
#             #     ind_xy = np.append(ind_x, ind_y)
#             #     y_2_npy = np.load('hotness_gpu_2.npy') if os.path.isfile("hotness_gpu_2.npy") else []
#             #     np.save('hotness_gpu_2.npy', np.append(y_2_npy, ind_xy))

#             # if indices_[0].get_device() == 3:
#             #     ind_x, ind_y = indices_[0].detach().cpu().numpy(), indices_[1].detach().cpu().numpy()
#             #     ind_xy = np.append(ind_x, ind_y)
#             #     y_3_npy = np.load('hotness_gpu_3.npy') if os.path.isfile("hotness_gpu_3.npy") else []
#             #     np.save('hotness_gpu_3.npy', np.append(y_3_npy, ind_xy))

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
#         self.a2a_cpu_time_ms = 0
#         self.a2a_cuda_event_intervals = []
#         if self.all2all_size > 1:
#             dispatched_input = self.all_to_all_wrapper(dispatched_input, is_expert=True)

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

#         # self.record_all_to_all_stats()
#         # print("entropy", self.metadata["entropy_gating"])
#         # print("number of expert", self.metadata["number of expert"])

#         return combined_output, l_aux

#     def prepare_for_inference_(self):
#         self.in_generation = True

#     def all_to_all_wrapper(self, input: Tensor, is_expert=False):
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
#         if is_expert:
#             output = _AllToAll_int8.apply(self.all2all_group, input)
#         else:
#             output = _AllToAll_fp.apply(self.all2all_group, input)
#         # newly added
#         cuda_end.record()
#         torch.cuda.synchronize()
#         print("New cuda time:",cuda_start.elapsed_time(cuda_end))
#         cpu_end = time.time() * 1000
#         self.a2a_cpu_time_ms += (cpu_end - cpu_start)
#         self.a2a_cuda_event_intervals.append((cuda_start, cuda_end))
#         return output

#     def record_all_to_all_stats(self):
#         # controlled via an argument as we want to minimize any impact from torch.cuda.synchronize()
#         # record_a2a_perf_stats = getattr(self.args, 'record_a2a_perf_stats', False)
#         # if "all_to_all_cpu_time_ms" not in self.metadata:
#         #     self.metadata["all_to_all_cpu_time_ms"] = 0
#         #     self.metadata["all_to_all_cuda_time_ms"] = 0
#         record_a2a_perf_stats = True
#         if record_a2a_perf_stats:
#             torch.cuda.synchronize()
#             self.total_cpu += self.a2a_cpu_time_ms

#             a2a_cuda_time_ms = 0.0
#             for ev_start, ev_end in self.a2a_cuda_event_intervals:
#                 a2a_cuda_time_ms += ev_start.elapsed_time(ev_end)
#             self.total_gpu += a2a_cuda_time_ms
#         # reset stats
#             self.a2a_cpu_time_ms = 0.0
#             self.a2a_cuda_event_intervals = []
    
            
#             logger.info(
#                 f'cpu time: {self.total_cpu}, gpu time: {self.total_gpu}'
#             )
#             logger.info(
#             'Expert1 overflow: {}\n'
#             'Expert2 overflow: {}\n'
#             'Expert1 unused count: {}\n'
#             'Expert2 unused count: {}\n'
#             'Expert1 balance top: {}\n'
#             'Expert2 balance top: {}\n'.format(
#                 self.metadata["overflow_expert1"],
#                 self.metadata["overflow_expert2"],
#                 self.metadata["unused_expert1_count"],
#                 self.metadata["unused_expert2_count"],
#                 self.metadata["expert1_balance_top"],
#                 self.metadata["expert2_balance_top"]
#             )
#             )