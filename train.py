import os
import sys
import hydra
import logging
import evaluate
import numpy as np
from dataclasses import dataclass
from typing import Optional
from datasets import load_dataset
from huggingface_hub import HfApi
from omegaconf import DictConfig
from transformers import TrainingArguments, set_seed, AutoTokenizer, AutoModelForSequenceClassification, Trainer, DataCollatorWithPadding
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str]
    lang: Optional[str]
    overwrite_cache: bool
    validation_split_percentage: Optional[int]
    max_seq_length: Optional[int]
    preprocessing_num_workers: Optional[int]
    mlm_probability: float
    line_by_line: bool
    pad_to_max_length: bool
    max_train_samples: Optional[int]
    max_eval_samples: Optional[int]

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str]
    tokenizer_name: Optional[str]
    cache_dir: Optional[str]
    model_revision: str
    use_fast_tokenizer: bool
    use_auth_token: bool

@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config: DictConfig, **kwargs):

    # Set up training arguments
    training_args = TrainingArguments(
        hub_token=os.environ.get("HUGGINGFACE_HUB_TOKEN", True),
        **config.training
    )

    # Set up data arguments
    data_args = DataArguments(**config.data)

    # Set up model arguments
    model_args = ModelArguments(**config.model)

    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    #Login to HuggingFace Hub
    api = HfApi(token=os.environ.get("HUGGINGFACE_HUB_TOKEN", True))

    # Create a repository on HuggingFace Hub
    api.create_repo(
        "ajders/ddisco_classifier",
        exist_ok=True,
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            cache_dir=model_args.cache_dir if model_args.cache_dir != "None" else None,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        raise ValueError(
                f"No dataset provided... config.data.dataset_name: ({data_args.dataset_name})."
            )
    
    # Rename column
    raw_datasets = raw_datasets.rename_columns({"rating": "labels"})

    # Reduce label index by 1 so they are 0, 1, 2
    raw_datasets = raw_datasets.map(lambda x: {"labels": x["labels"] - 1})

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir if model_args.cache_dir != "None" else None,
        use_fast=model_args.use_fast_tokenizer,
    )

    # Define tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding=data_args.pad_to_max_length,
            truncation=True,
            max_length=data_args.max_seq_length,
            return_tensors="pt",
        )

    # tokenizing the dataset
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=3,
        revision=model_args.model_revision,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        cache_dir=model_args.cache_dir if model_args.cache_dir != "None" else None,
    )

    # Define compute metrics function
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"] if training_args.do_train else None,
        eval_dataset=tokenized_datasets["test"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_result = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in eval_result.items():
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}")
    
    # Push to HuggingFace Hub
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
