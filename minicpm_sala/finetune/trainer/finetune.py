# -*- coding: utf-8 -*-
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="openbmb/MiniCPM-2B-sft-bf16")


@dataclass
class DataArguments:
    train_data_path: str = field(
        default="data/AdvertiseGenChatML/train.json",
        metadata={"help": "Path to the training data."},
    )
    eval_data_path: str = field(
        default="data/AdvertiseGenChatML/dev.json",
        metadata={"help": "Path to the test data."},
    )


@dataclass
class CustomTrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be truncated."},
    )
    use_lora: bool = field(default=True)
    qlora: bool = field(default=False)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning with ChatML schema."""

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        model_max_length: int = 4096,
    ):
        super(SupervisedDataset, self).__init__()
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.ignore_index = -100

        # Sanity check logging
        if len(self.data) > 0:
            sample = self.preprocessing(self.data[0])
            print("\n--- Preprocessing Debug Sample ---")
            print("Decoded Input:", self.tokenizer.decode(sample["input_ids"]))
            valid_labels = [l for l in sample["labels"] if l != self.ignore_index]
            print("Decoded Labels:", self.tokenizer.decode(valid_labels))
            print("-----------------------------------\n")

    def __len__(self):
        return len(self.data)

    def preprocessing(self, example: dict) -> Dict[str, torch.Tensor]:
        input_ids = []
        label_ids = []

        if self.tokenizer.bos_token_id is not None:
            input_ids.append(self.tokenizer.bos_token_id)
            label_ids.append(self.ignore_index)

        is_minicpm3_or_4 = getattr(self.tokenizer, "eos_token_id", None) == 73440

        for message in example.get("messages", []):
            role = message.get("role")
            content = message.get("content", "")

            if role in ["user", "system"]:
                if is_minicpm3_or_4 and role == "user":
                    formatted_ids = self.tokenizer.apply_chat_template(
                        [message], add_generation_prompt=True, tokenize=True
                    )
                else:
                    formatted_ids = self.tokenizer.apply_chat_template(
                        [message], tokenize=True
                    )

                input_ids.extend(formatted_ids)
                label_ids.extend([self.ignore_index] * len(formatted_ids))

            elif role == "assistant":
                if is_minicpm3_or_4:
                    response_ids = self.tokenizer.encode(content, add_special_tokens=False)
                else:
                    response_ids = self.tokenizer.apply_chat_template(
                        [message], tokenize=True
                    )

                input_ids.extend(response_ids)
                label_ids.extend(response_ids)

        if self.tokenizer.eos_token_id is not None:
            input_ids.append(self.tokenizer.eos_token_id)
            label_ids.append(self.tokenizer.eos_token_id)

        # Truncate to maximum length
        input_ids = input_ids[: self.model_max_length]
        label_ids = label_ids[: self.model_max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.preprocessing(self.data[idx])


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate and dynamically pad examples for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]

        # Dynamically pad inputs to longest in batch
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }


def load_model_and_tokenizer(
    model_path: str,
    use_lora: bool = True,
    qlora: bool = False,
    bf16: bool = False,
    fp16: bool = False,
):
    """Loads model, tokenizer, and configures LoRA/QLoRA setup."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)

    quantization_config = None
    if qlora:
        assert use_lora, "use_lora must be True when qlora is enabled."
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        quantization_config=quantization_config,
    )

    if qlora:
        model = prepare_model_for_kbit_training(model)

    if use_lora:
        target_modules = (
            ["q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "kv_b_proj"]
            if getattr(model.config, "architectures", [""])[0] == "MiniCPM3ForCausalLM"
            else ["q_proj", "v_proj"]
        )

        lora_config = LoraConfig(
            init_lora_weights="gaussian",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            r=64,
            lora_alpha=32,
            lora_dropout=0.1,
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, CustomTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model, tokenizer = load_model_and_tokenizer(
        model_path=model_args.model_name_or_path,
        use_lora=training_args.use_lora,
        qlora=training_args.qlora,
        bf16=training_args.bf16,
        fp16=training_args.fp16,
    )

    train_dataset = SupervisedDataset(
        data_path=data_args.train_data_path,
        tokenizer=tokenizer,
        model_max_length=training_args.model_max_length,
    )
    eval_dataset = SupervisedDataset(
        data_path=data_args.eval_data_path,
        tokenizer=tokenizer,
        model_max_length=training_args.model_max_length,
    )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
