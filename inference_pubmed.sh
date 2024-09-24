DATA_PATH=/home/vpkmiriyal/PILE/pile-pubmedabstract/data-bin
MODEL_PATH=/home/vpkmiriyal/model_data/en_moe_lm_15b/model.pt

rm -rf *.npy #clear previous output files

python -m fairseq_cli.eval_lm $DATA_PATH \
  --ddp-backend fully_sharded \
  --path $MODEL_PATH \
  --fp16 \
  --max-valid-steps 100 \
  --batch-size 1 \
  --gen-subset valid \
  --bpe gpt2 \
  --softmax-batch 1024 \
  --tokens-per-sample 1024 \
  --sample-break-mode none \
  --is-moe \
  --distributed-world-size 4 \
  --model-overrides "{'world_size': 4, 'moe_eval_capacity_token_fraction': 1}"

python post_process.py --dataset Pubmed --file_name hotness_gpu_0.npy
