import argparse

from datasets import Dataset, concatenate_datasets
from transformers import (Trainer, TrainingArguments, Wav2Vec2ForPreTraining,
                          Wav2Vec2Processor)

from collators import PretrainCollator


def make_dataset(dataset_name, proc_dir):
    dataset = Dataset.load_from_disk(f"datafast/{dataset_name}.train.trim_12.hf")
    processor = Wav2Vec2Processor.from_pretrained(proc_dir)
    return dataset, processor


def make_model(processor):
    model = Wav2Vec2ForPreTraining.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53",
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.4,
        layerdrop=0.1,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )
    model = model.cuda()
    model.freeze_feature_extractor()
    model.gradient_checkpointing_enable()
    return model


def make_training_args(dataset_name, args):
    training_args = TrainingArguments(
        output_dir=f"./xlsr-{dataset_name}",
        group_by_length=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        max_steps=args.max_steps,
        fp16=True,
        save_steps=args.max_steps // 2,
        logging_steps=10,
        learning_rate=3e-4,
        warmup_steps=int(0.1 * args.max_steps),
        save_total_limit=2,
    )
    return training_args


def parse_args():
    parser = argparse.ArgumentParser("Domain adaptive pretraining")
    parser.add_argument("--dataset-name", type=str, help="dataset to run copt on")
    parser.add_argument("--batch-size", type=int, default=4, help="batch size")
    parser.add_argument("--max-steps", type=int, default=20000, help="max steps")
    parser.add_argument(
        "--resume", type=str, default=None, help="Checkpoint to resume training"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    processor_dir = "processor/"
    train_dataset, processor = make_dataset(args.dataset_name, processor_dir)
    model = make_model(processor)
    data_collator = PretrainCollator(processor, model_config=model.config)

    training_args = make_training_args(args.dataset_name, args)
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=processor.feature_extractor,
    )

    trainer.train(resume_from_checkpoint=args.resume is not None)


if __name__ == "__main__":
    main()
