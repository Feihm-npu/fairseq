#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from tutel import system, net

# Initialize parallel environment using NCCL backend
parallel_env = system.init_data_model_parallel(backend='nccl', group_count=1)
local_device = parallel_env.local_device

# Assert that the world size is 2
assert parallel_env.global_size == 2, "This test case is set for World Size == 2 only"

# Function to measure transfer time
def measure_transfer_time(input, send_counts):
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    output = net.batch_all_to_all_v([input,], send_counts)[0]
    end_time.record()

    torch.cuda.synchronize()
    transfer_time = start_time.elapsed_time(end_time)  # in milliseconds
    return transfer_time, output

# Set input tensors and send counts based on global rank
if parallel_env.global_rank == 0:
    input_float16 = torch.tensor(range(102400), dtype=torch.float16, device=local_device)
    input_int8 = torch.tensor(range(102400), dtype=torch.int8, device=local_device)
    send_counts = torch.tensor([0, 102400], dtype=torch.int64, device=local_device)
else:
    input_float16 = torch.tensor(range(102400), dtype=torch.float16, device=local_device)
    input_int8 = torch.tensor(range(102400), dtype=torch.int8, device=local_device)
    send_counts = torch.tensor([0, 102400], dtype=torch.int64, device=local_device)

# Synchronize all processes
net.barrier()

# Measure transfer time for float16 tensor
# transfer_time_float16, output_float16 = measure_transfer_time(input_float16, send_counts)
# print(f'Device-{parallel_env.global_rank} recvs float16: cuda{output_float16}')
# print(f'Device-{parallel_env.global_rank} transfer time for float16 tensor: {transfer_time_float16} ms')

# Measure transfer time for int8 tensor
transfer_time_int8, output_int8 = measure_transfer_time(input_int8, send_counts)
print(f'Device-{parallel_env.global_rank} recvs int8: {output_int8}')
print(f'Device-{parallel_env.global_rank} transfer time for int8 tensor: {transfer_time_int8} ms')


