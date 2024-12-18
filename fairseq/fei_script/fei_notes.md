# Fairseq benchmark
- Steps to run fairseq code on the cluster:
    1. Allocate an interactive GPU node with the following command: salloc -N 1 -n 8 -p mi1008x -t 24:00:00
    2. Creating a python virtual env: python -m venv .venv
    3. Activate the environment: source .venv/bin/activate
    4. Following the docker steps:
        - Torch:
            - pip uninstall -y torch torchvision torchaudio
            - pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
        - Fairseq dependencies:
            - pip install fairscale==0.4.0
            - pip install hydra-core==1.0.7 omegaconf==2.0.6
        - Tutel:
            - pip uninstall -y tutel
            - git clone https://github.com/microsoft/tutel --branch main
            - sed -i '1s/^/#define __HIP_PLATFORM_HCC__ /' tutel/tutel/custom/custom_kernel.cpp
            - python ./tutel/setup.py install
        - Apex:
            - pip uninstall -y apex
            - pip uninstall -y apex (to run twice to be sure)
            - git clone https://github.com/ROCm/apex.git
            - git checkout release/1.2.0
            - python setup.py install --cpp_ext --cuda_ext
        - pip install iopath pyarrow pandas argparse matplotlib
        - Flash attention (never tried but could speedup the model):
            - Follow instruction on https://github.com/ROCm/flash-attention

## Model run related
- Control the number of threads: export OMP_NUM_THREADS=4
- pip install mkl-service==2.4.0 according to https://github.com/pytorch/pytorch/issues/37377

- CUDA_VISIBLE_DEVICES=0,1 rocprof --stats -o ./gpu2-1.csv fairseq-train data-bin/iwslt14.tokenized.de-en     --optimizer nag --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000     --arch fconv_iwslt_de_en --save-dir checkpoints/fconv --max-epoch 6 --ddp-backend fully_sharded --fp16



## Fairseq evalation on capacity
```bash
#!/bin/bash
module restore
source /work1/amd/hongmfei/moespace/.moefair/bin/activate
cd /work1/amd/hongmfei/moespace/SPEED-main/fairseq

DATA_PATH=/work1/amd/hongmfei/raw_data/data-bin/wiki/
MODEL_PATH=/work1/amd/hongmfei/models/en_moe_lm_15b/model.pt
WORLD_SIZE=4
export OMP_NUM_THREADS=4

rm -rf *.npy
RESULTS_FILE="evaluation_results_73.csv"
echo "capacity,it/s,tokens/s,valid_loss,perplexity" > $RESULTS_FILE
for capacity in $(seq 0.5 0.01 0.51); do
  OUTPUT_FILE="output_${capacity}.log"
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
    --distributed-world-size $WORLD_SIZE \
    --seed 100 \
    --model-overrides "{'world_size': $WORLD_SIZE, 'moe_eval_capacity_token_fraction': $capacity}" \
    &> $OUTPUT_FILE
  evaluated_line=$(grep "Evaluated" $OUTPUT_FILE)
  tokens=$(echo $evaluated_line | grep -Po "(?<=Evaluated )\d+")
  total_time=$(echo $evaluated_line | grep -Po "(?<=tokens in )\d+\.\d+")
  tokens_per_s=$(echo $evaluated_line | grep -Po "(?<=\()\d+\.\d+(?= tokens/s)")
  valid_loss=$(grep -Po "(?<=valid Loss \(base 2\): )\d+\.\d+" $OUTPUT_FILE)
  perplexity=$(grep -Po "(?<=Perplexity: )\d+\.\d+" $OUTPUT_FILE)
  echo "$capacity,$tokens,$total_time,$tokens_per_s,$valid_loss,$perplexity" >> $RESULTS_FILE
  echo "$capacity completed"
  rm $OUTPUT_FILE
done
echo "All Steps completed."
```

## Fairseq communication quantization experiments
```bash
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
```


# Mixtral implementation
### Preparation
- Install editable version of the moe branch of Fairseq https://github.com/pytorch/fairseq/#requirements-and-installation
- transformers, huggingface datasets
    ```sh
    pip install datasets
    git clone https://github.com/huggingface/transformers.git
    cd transformers
    pip install -e .
    ```

## Mixtral and Qwen evaluation
1. Replace the modeling_mixtral.py file @ transformers/src/transformers/models/mixtral/
2. Repalce the modeling_qwen2_moe.py file @ transformers/src/transformers/models/qwen2_moe/
3. Install lm-eval-harness
    ```bash
    git clone https://github.com/EleutherAI/lm-evaluation-harness.git
    cd lm-evaluation-harness
    pip install -e .
    ```
4. Evaluate the quantization method
    - Mixtral
    ```bash
    #!/bin/bash

    module restore
    source /work1/amd/hongmfei/moespace/.moefair/bin/activate
    cd /work1/amd/hongmfei/moespace/lm-evaluation-harness
    LOG_DIR="$WORK"

    # export PYTORCH_ROCM_ARCH="gfx1031"
    # export HSA_OVERRIDE_GFX_VERSION=10.3.1
    export AMD_SERIALIZE_KERNEL=3
    export TORCH_USE_HIP_DSA=1

    export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
    export OMP_NUM_THREADS=64
    TASKS=lambada_multilingual_stablelm
    ## Qmode 0: origin 1: hot 2 all
    # Qmode=1
    # NumHotExperts=1
    ## Qprec int8 int4 fp8 fp8base
    # Qprec
    ## hellaswag,lambada_multilingual_stablelm,mmlu,gsm8k,lambada_multilingual_stablelm,wmt16
    ## lambada_openai_mt_stablelm_en
    for Qmode in 1; do
        for NumHotExperts in 1 2 3 4; do
            for Qpre in int7 int6 int5; do
                HIP_VISIBLE_DEVICES=0,1,2,3 lm_eval --model hf \
                    --model_args pretrained=mistralai/Mixtral-8x7B-Instruct-v0.1,parallelize=True,Qmode=$Qmode,NumHotExperts=$NumHotExperts,Qpre=$Qpre \
                    --tasks $TASKS \
                    --output_path "mixtra_{$TASKS}_{$Qmode}_{$NumHotExperts}.json" \
                    --device cuda \
                    --limit 100 \
                    --batch_size 64 &> "$LOG_DIR/lm-eval/mixtral_eval1218/mistral_{$TASKS}_Q{$Qmode}_hot{$NumHotExperts}_Qpref{$Qpre}.log"
                echo "Task Qmode $Qmode Number of Hot Experts $NumHotExperts Quantize precision $Qpre completed!"
            done
        done
    done
    ```
    - Qwen moe
    ```sh
    #!/bin/bash
    module restore
    source /work1/amd/hongmfei/moespace/.moefair/bin/activate
    cd /work1/amd/hongmfei/moespace/lm-evaluation-harness
    LOG_DIR="$WORK"
    export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
    export OMP_NUM_THREADS=64
    TASKS=hellaswag
    ## hellaswag,lambada_multilingual_stablelm,mmlu,gsm8k,lambada_multilingual_stablelm,wmt16,lambada_openai_mt_stablelm_en 
    ## monology/pile-uncopyrighted
    lm_eval --model hf \
        --model_args pretrained=Qwen/Qwen1.5-MoE-A2.7B-Chat \
        --tasks $TASKS \
        --output_path "qwen_{$TASKS}.json" \
        --device cuda \
        --batch_size auto:1 &> "$LOG_DIR/lm-eval/qwen_{$TASKS}_hot_int8.log"
    # ,parallelize=True
    ```

## Mixtral implementation
- A simple version can be find in [hf_mixtral.py](../models/huggingface/hf_mixtral.py) where a task is registered and import the mixtral model from huggingface api. However, this is a non-parallelism implementation.
- Expert parallelism implementation
1. Define the mixtral layers, architecture and task
    - [mixtral_layer.py](../models/mixtral_layer.py): all the functions are taken from transformers/src/transformers/models/mixtral/mixtral_modeling.py.
    - [mixtral.py](../models/mixtral.py): wrap the moe layers and the experts
    ```py
    def fsdp_wrap_expert(args, layer, min_num_params=0):
        process_group = layer.moe_layer.expert_group
        for i, expert in enumerate(layer.moe_layer.experts):
            layer.moe_layer.experts[i] = fsdp_wrap(expert, process_group=process_group, min_num_params=0)
        layer = fsdp_wrap(layer, min_num_params=min_num_params)
        return layer
    ```
    - [mixtral_lm.py](../models/mixtral_lm.py):
    register the model and architechture
    ```py
    @register_model("mixtral_lm", dataclass=MixtralLanguageModelingConfig)
    class MixtralLMModel(FairseqLanguageModel):
        pass
    @register_model_architecture("mixtral_lm", "mixtral_lm_arch")
    def mixtral_lm_arch(args):
        pass
    ```
    - [mixtral_language_modeling.py](../tasks/mixtral_language_modeling.py)
    ```py
    @register_task("mixtral_language_modeling", dataclass=MixtralLanguageModelingConfig)
    class MixtralLanguageModelingTask(LanguageModelingTask):
        pass
    ### load the tokenizer to get the word dictionary
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name)
    vocab = tokenizer.get_vocab()
    dictionary = Dictionary()
    ```
2. train the model
    ```bash
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
    ```
3. inference
    ```bash
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
    ```
    
### For reference -- Transformer Structure for Language Models
- [tranformer_lm.py](../models/transformer_lm.py) --> [transformer.py](../models/transformer.py) --> [transformer_layer.py](../modules/transformer_layer.py)
- [tranformer_lm.py](../models/transformer_lm.py) registers the model and model structure
    - **class transfoermerLanguageModel()** build a new model instance
    ```python
    decoder = TransformerDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True,
        )
    ```
    - different architecture initialize different hyperparameters
- [transformer.py](../models/transformer.py)
    - build the decoder layer and wrap with fsdp **fsdp_wrap_expert**

- [transformer_layer.py](../modules/transformer_layer.py)
    - Define the **TransformerDecoderLayer** inherented from **nn.Module**

