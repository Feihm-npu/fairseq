参考我的笔记内容，将mixtral模型包装在fairseq框架里。fsdp_wrap也需要实现，gate和expert可以使用fairseq.modules.moe的实现Top1Gate, Top2Gate, MOELayer：。我期待输出三个python文件，1. mixtral_lm.py 负责注册模型和模型架构；2. mixtral.py 负责构建模型；3. mixtral_layer.py，负责定义模型。与刚才给你的三个py程序一一对应。
To understand the fairseq framework, the notes are made.
### Transformer Structure for Language Models
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


进一步的要求：
1. 输入参数中仅仅保留