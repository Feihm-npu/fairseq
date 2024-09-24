DATA_PATH=/work1/amd/hongmfei/raw_data/data-bin/wiki/
MODEL_PATH=/work1/amd/hongmfei/models/en_moe_lm_15b/model.pt

rm -rf *.npy #clear previous output files

python -m fairseq_cli.eval_lm $DATA_PATH \
  --ddp-backend fully_sharded \
  --path $MODEL_PATH \
  --fp16 \
  --max-valid-steps 100 \
  --batch-size 1 \
  --gen-subset valid \
  --bpe gpt2 \
  --softmax-batch 2048 \
  --tokens-per-sample 2048 \
  --sample-break-mode none \
  --is-moe \
  --distributed-world-size 4 \
  --seed 100 \
  --model-overrides "{'world_size': 4, 'moe_eval_capacity_token_fraction': 0.5}"

#python post_process.py --dataset wiki --file_name hotness_gpu_0.py

