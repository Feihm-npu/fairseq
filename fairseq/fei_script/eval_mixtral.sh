module restore
source /work1/amd/hongmfei/moespace/.moefair/bin/activate
cd /work1/amd/hongmfei/moespace/SPEED-main/fairseq
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

## To solve the missing module bugs
FAIRSEQ_DIR=$(pip list -v | grep 'fairseq' | awk '{print $3}')
export PYTHONPATH=$PYTHONPATH:$FAIRSEQ_DIR

DATA_PATH=/work1/amd/hongmfei/raw_data/data-bin/wiki/
MODEL_PATH=/work1/amd/hongmfei/moespace/SPEED-main/fairseq/checkpoints-51200/checkpoint_last.pt
python -m fairseq_cli.eval_lm \
  $DATA_PATH \
  --path $MODEL_PATH \
  --ddp-backend fully_sharded \
  --gen-subset valid \
  --sample-break-mode none \
  --tokens-per-sample 2048 \
  --batch-size 100 \
  --softmax-batch 2048 \
  --max-valid-steps 10 \
  --fp16 \
  --is-moe \
  --distributed-world-size 4 \
  --model-overrides "{'world_size': 4, 'moe_eval_capacity_token_fraction': 0.05, 'dict-size': 50000}"

    # --output-word-probs \

# from fairseq.checkpoint_utils import load_checkpoint_to_cpu
# ckpt_path = '/work1/amd/hongmfei/moespace/SPEED-main/fairseq/checkpoints-51200-lm/checkpoint_last-shared.pt'
# state = load_checkpoint_to_cpu(ckpt_path, is_moe=True)
