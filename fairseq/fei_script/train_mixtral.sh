module restore
source /work1/amd/hongmfei/moespace/.moefair/bin/activate
cd /work1/amd/hongmfei/moespace/SPEED-main/fairseq
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

## To solve the missing module bugs
FAIRSEQ_DIR=$(pip list -v | grep 'fairseq' | awk '{print $3}')
export PYTHONPATH=$PYTHONPATH:$FAIRSEQ_DIR
DATA_PATH=/work1/amd/hongmfei/raw_data/data-bin/wiki/

NUM_EXPERTS=4
TOKENS_PER_SAMPLE=2048
python fairseq_cli/train.py \
  --ddp-backend fully_sharded --memory-efficient-fp16 --checkpoint-activations \
  $DATA_PATH \
  --task language_modeling --tokens-per-sample $TOKENS_PER_SAMPLE \
  --arch mixtral_v1 --share-decoder-input-output-embed \
  --decoder-layers 32 --decoder-embed-dim 4096 --decoder-ffn-embed-dim 14336 \
  --decoder-attention-heads 64 \
  --moe-expert-count $NUM_EXPERTS --moe-freq 1 \
  --moe-gating-use-fp32 --moe-second-expert-policy all \
  --moe-normalize-expert-grad sqrt_world_size \
  --moe-eval-capacity-token-fraction -1.0 \
  --max-sentences-valid 1 --num-workers-valid 0 \
  --criterion moe_cross_entropy --moe-gate-loss-wt 0.01 --moe-gate-loss-combine-method sum \
  --optimizer adam --fp16-adam-stats --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr 0.0005 --warmup-updates 750 \
  --dropout 0.1 --attention-dropout 0.1 \
  --batch-size 4 --update-freq 1 \
  --max-update 1000 --disable-validation \
  --log-format json --log-interval 10 \
  --save-dir /work1/amd/hongmfei/moespace/SPEED-main/fairseq/checkpoints-mixtralv1 \
  --restore-file /work1/amd/hongmfei/moespace/SPEED-main/fairseq/checkpoints-mixtralv1/checkpoint_last.pt
# ```
# MixtralForCausalLM(
#   (model): MixtralModel(
#     (embed_tokens): Embedding(32000, 4096)
#     (layers): ModuleList(
#       (0-31): 32 x MixtralDecoderLayer(
#         (self_attn): MixtralSdpaAttention(
#           (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
#           (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
#           (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
#           (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
#           (rotary_emb): MixtralRotaryEmbedding()
#         )
#         (block_sparse_moe): MixtralSparseMoeBlock(
#           (gate): Linear(in_features=4096, out_features=8, bias=False)
#           (experts): ModuleList(
#             (0-7): 8 x MixtralBlockSparseTop2MLP(
#               (w1): Linear(in_features=4096, out_features=14336, bias=False)
#               (w2): Linear(in_features=14336, out_features=4096, bias=False)
#               (w3): Linear(in_features=4096, out_features=14336, bias=False)
#               (act_fn): SiLU()
#             )
#           )
#         )
#         (input_layernorm): MixtralRMSNorm((4096,), eps=1e-05)
#         (post_attention_layernorm): MixtralRMSNorm((4096,), eps=1e-05)
#       )
#     )
#     (norm): MixtralRMSNorm((4096,), eps=1e-05)
#   )
#   (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
# )
# ```