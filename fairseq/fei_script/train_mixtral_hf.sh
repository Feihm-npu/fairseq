#!/usr/bin/env bash

module restore

source /work1/amd/hongmfei/moespace/.moefair/bin/activate

cd /work1/amd/hongmfei/moespace/SPEED-main/fairseq

export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# To get the fairseq path
FAIRSEQ_DIR=$(pip list -v | grep 'fairseq' | awk '{print $3}')
export PYTHONPATH=$PYTHONPATH:$FAIRSEQ_DIR

DATA_PATH=/work1/amd/hongmfei/raw_data/data-bin/wiki/

NUM_EXPERTS=4
TOKENS_PER_SAMPLE=2048


python fairseq_cli/train.py \
  --task mixtral_language_modeling --hf-model-name mistralai/Mixtral-8x7B-v0.1 \
  $DATA_PATH \
  --tokens-per-sample $TOKENS_PER_SAMPLE \
  --sample-break-mode none \
  --arch mixtral_lm_arch --share-decoder-input-output-embed \
  --decoder-layers 2 \
  --decoder-embed-dim 4096 \
  --decoder-ffn-embed-dim 14336 \
  --decoder-attention-heads 64 \
  --moe-expert-count $NUM_EXPERTS \
  --moe-freq 1 \
  --moe-gating-use-fp32 \
  --moe-second-expert-policy all \
  --moe-normalize-expert-grad sqrt_world_size \
  --moe-eval-capacity-token-fraction -1.0 \
  --max-sentences-valid 1 \
  --num-workers-valid 0 \
  --criterion moe_cross_entropy \
  --moe-gate-loss-wt 0.01 \
  --moe-gate-loss-combine-method sum \
  --optimizer adam --fp16-adam-stats --adam-betas '(0.9, 0.98)' \
  --clip-norm 0.0 \
  --lr 0.0005 \
  --warmup-updates 750 \
  --dropout 0.1 \
  --attention-dropout 0.1 \
  --batch-size 4 \
  --update-freq 1 \
  --max-update 10 \
  --disable-validation \
  --log-format json \
  --log-interval 10 \
  --save-dir /work1/amd/hongmfei/moespace/SPEED-main/fairseq/checkpoints-mixtralv2 \
  --restore-file /work1/amd/hongmfei/moespace/SPEED-main/fairseq/checkpoints-mixtralv2/checkpoint_last.pt \
  --ddp-backend fully_sharded --memory-efficient-fp16 --checkpoint-activations
