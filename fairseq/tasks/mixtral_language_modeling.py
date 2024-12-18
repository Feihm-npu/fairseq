import logging
import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch

from transformers import AutoTokenizer

from fairseq import utils
from fairseq.data import (
    AppendTokenDataset,
    Dictionary,
    IdDataset,
    LMContextWindowDataset,
    NestedDictionaryDataset,
    MonolingualDataset,
    NumelDataset,
    PadDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TokenBlockDataset,
    TruncatedDictionary,
    data_utils,
)
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.tasks.language_modeling import LanguageModelingTask, LanguageModelingConfig

logger = logging.getLogger(__name__)

@dataclass
class MixtralLanguageModelingConfig(LanguageModelingConfig):
    hf_model_name: str = field(
        default="mistralai/Mixtral-8x7B-v0.1",
        metadata={"help": "Name or path of the Hugging Face Mixtral model and tokenizer"}
    )

@register_task("mixtral_language_modeling", dataclass=MixtralLanguageModelingConfig)
class MixtralLanguageModelingTask(LanguageModelingTask):
    """
    基于Hugging Face上的Mixtral模型与tokenizer的语言模型任务。
    该任务与language_modeling类似，但从HF的tokenizer中构建Dictionary。
    """

    @classmethod
    def setup_dictionary(cls, args, **kwargs):
        """
        使用Hugging Face的Tokenizer来构建Fairseq的Dictionary。
        """
        if not args.hf_model_name:
            raise ValueError("Must provide hf_model_name for Mixtral tokenizer")

        # 加载HF tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name)
        # 获取词表
        vocab = tokenizer.get_vocab()

        # 构建 Fairseq Dictionary
        dictionary = Dictionary()
        # vocab是{token: idx}，需要按index顺序插入token确保与tokenizer一致
        # 注意：get_vocab()通常返回按词频排序的vocab，但为了安全，我们根据idx排序
        inv_vocab = sorted(vocab.items(), key=lambda x: x[1])

        for token, idx in inv_vocab:
            dictionary.add_symbol(token)
        dictionary.finalize()

        output_dictionary = dictionary
        if args.output_dictionary_size >= 0:
            output_dictionary = TruncatedDictionary(
                dictionary, args.output_dictionary_size
            )

        logger.info("dictionary: {} types".format(len(dictionary)))
        return (dictionary, output_dictionary)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """
        覆盖setup_task以便使用Mixtral tokenizer构建dictionary。
        """
        dictionary, output_dictionary = cls.setup_dictionary(args, **kwargs)

        # upgrade old checkpoints
        if getattr(args, "exclude_self_target", False):
            args.self_target = False

        targets = []
        if getattr(args, "self_target", False):
            targets.append("self")
        if getattr(args, "future_target", False):
            targets.append("future")
        if getattr(args, "past_target", False):
            targets.append("past")
        if len(targets) == 0:
            # standard language modeling
            targets = ["future"]

        return cls(args, dictionary, output_dictionary, targets=targets)

    def load_dataset(
        self, split: str, epoch=1, combine=False, **kwargs
    ) -> MonolingualDataset:
        """
        加载数据集split并进行TokenBlock处理。
        假设您的原始数据已通过Fairseq预处理为indexed dataset形式。
        如果您需要用HF tokenizer重新分词原始文本，则需要在预处理时进行。
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0

        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        dataset = data_utils.load_indexed_dataset(
            split_path, self.dictionary, self.args.dataset_impl, combine=combine
        )
        if dataset is None:
            raise FileNotFoundError(f"Dataset not found: {split} ({split_path})")

        dataset = maybe_shorten_dataset(
            dataset,
            split,
            self.args.shorten_data_split_list,
            self.args.shorten_method,
            self.args.tokens_per_sample,
            self.args.seed,
        )
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.args.tokens_per_sample,
            pad=self.dictionary.pad(),
            eos=self.dictionary.eos(),
            break_mode=self.args.sample_break_mode,
            include_targets=True,
            use_plasma_view=self.args.use_plasma_view,
            split_path=split_path,
            plasma_path=self.args.plasma_path,
        )

        add_eos_for_other_targets = (
            self.args.sample_break_mode is not None
            and self.args.sample_break_mode != "none"
        )
        fixed_pad_length = None
        if self.args.pad_to_fixed_length:
            fixed_pad_length = self.args.tokens_per_sample

        pad_to_bsz = None
        if self.args.pad_to_fixed_bsz:
            pad_to_bsz = self.args.batch_size_valid if 'valid' in split else self.args.batch_size

        self.datasets[split] = MonolingualDataset(
            dataset=dataset,
            sizes=dataset.sizes,
            src_vocab=self.dictionary,
            tgt_vocab=self.output_dictionary,
            add_eos_for_other_targets=add_eos_for_other_targets,
            shuffle=True,
            targets=self.targets,
            add_bos_token=self.args.add_bos_token,
            fixed_pad_length=fixed_pad_length,
            pad_to_bsz=pad_to_bsz,
        )

    def build_model(self, args):
        """
        在这里加载HF的Mistral模型（AutoModelForCausalLM）。
        您需要在model文件中实现对应的Fairseq模型类（例如MixtralLMModel），
        以适配Fairseq框架，并在build_model中实例化它。
        """
        model = super().build_model(args)
        for target in self.targets:
            if target not in model.supported_targets:
                raise ValueError(
                    f"Unsupported language modeling target: {target} not in {model.supported_targets}"
                )
        return model

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        """
        为推理构建数据集，与language_modeling任务类似。
        这里假设src_tokens已经是tensor形式的token序列。
        """
        dataset = StripTokenDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                block_size=None,  # "eos"模式下此参数无效
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            # 移除目标序列末尾的eos
            self.source_dictionary.eos(),
        )
        src_dataset = PrependTokenDataset(
            dataset,
            token=(
                self.source_dictionary.bos()
                if getattr(self.args, "add_bos_token", False)
                else self.source_dictionary.eos()
            ),
        )
        tgt_dataset = AppendTokenDataset(dataset, token=self.source_dictionary.pad())
        return NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": PadDataset(
                        src_dataset,
                        pad_idx=self.source_dictionary.pad(),
                        left_pad=False,
                    ),
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                },
                "target": PadDataset(
                    tgt_dataset, pad_idx=self.source_dictionary.pad(), left_pad=False
                ),
            },
            sizes=[np.array(src_lengths)],
        )

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        """
        基于给定的样本和模型进行推理，与language_modeling任务的实现类似。
        """
        with torch.no_grad():
            if getattr(self.args, "add_bos_token", False):
                bos_token = self.source_dictionary.bos()
            else:
                bos_token = self.source_dictionary.eos()

            if constraints is not None:
                raise NotImplementedError(
                    "Constrained decoding with the mixtral_language_modeling task is not supported"
                )

            if prefix_tokens is None and sample["net_input"]["src_tokens"].nelement():
                prefix_tokens = sample["net_input"]["src_tokens"]
                if prefix_tokens[:, 0].eq(bos_token).all():
                    prefix_tokens = prefix_tokens[:, 1:]

            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, bos_token=bos_token
            )

    def eval_lm_dataloader(
        self,
        dataset,
        max_tokens: Optional[int] = 36000,
        batch_size: Optional[int] = None,
        max_positions: Optional[int] = None,
        num_shards: int = 1,
        shard_id: int = 0,
        num_workers: int = 1,
        data_buffer_size: int = 10,
        context_window: int = 0,
    ):
        """
        为语言模型评估构建dataloader，类似于language_modeling中的eval_lm_dataloader实现。
        """
        if context_window > 0:
            dataset = LMContextWindowDataset(
                dataset=dataset,
                tokens_per_sample=self.args.tokens_per_sample,
                context_window=context_window,
                pad_idx=self.source_dictionary.pad(),
            )
        return self.get_batch_iterator(
            dataset=dataset,
            max_tokens=max_tokens,
            max_sentences=batch_size,
            max_positions=max_positions,
            ignore_invalid_inputs=True,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            data_buffer_size=data_buffer_size,
        ).next_epoch_itr(shuffle=False)
