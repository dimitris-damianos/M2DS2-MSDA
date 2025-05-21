from datasets import Dataset, disable_caching, load_metric
from transformers import (EarlyStoppingCallback, IntervalStrategy, Trainer,
                          TrainingArguments, Wav2Vec2ForCTC, Wav2Vec2Processor)

from collators import CTCCollator
from utils import (get_pretrained_model, load_model_from_pretrained,
                   make_metrics_calculator, make_parser)

disable_caching()


def make_dataset(source_name, target_name, silver_labels, processor, max_seconds=12):
    with open(silver_labels, "r") as fd:
        lines = [ln.strip() for ln in fd]
        utt_ids = [ln.split()[0] for ln in lines]
        sentences = [" ".join(ln.split()[1:]) + " " for ln in lines]
        sentence_dict = dict(zip(utt_ids, sentences))

    def map_silver_labels(batch):
        batch["sentence"] = sentence_dict[batch["utt_id"]]
        with processor.as_target_processor():
            batch["labels"] = processor(batch["sentence"]).input_ids
        return batch

    dataset = Dataset.load_from_disk(
        f"datafast/{target_name}.train.trim_{max_seconds}.hf"
    )
    dataset = dataset.map(map_silver_labels)
    target_dev_dataset = Dataset.load_from_disk(f"datafast/{target_name}.dev.hf")
    test_dataset = Dataset.load_from_disk(f"datafast/{target_name}.test.hf")

    print(
        f"LOOOOG: Finetuning on {target_name} target domain with pseudolabels from {source_name} domain"
    )

    return dataset, target_dev_dataset, test_dataset


def make_model(processor, args):
    pretrained_model = get_pretrained_model(
        args.target_name, copt=args.copt, resume=args.resume
    )
    model = load_model_from_pretrained(processor, Wav2Vec2ForCTC, pretrained_model)
    return model


def make_training_args(source_name, target_name, args):

    output_dir = f"./xlsr-pseudolabel-src_{source_name}-tgt_{target_name}"
    training_args = TrainingArguments(
        # output_dir="/content/gdrive/MyDrive/wav2vec2-large-xlsr-turkish-demo",
        output_dir=output_dir,
        group_by_length=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        evaluation_strategy=IntervalStrategy.STEPS,
        max_steps=args.max_steps,
        fp16=True,
        save_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        logging_steps=10,
        learning_rate=3e-4,
        warmup_steps=int(0.1 * args.max_steps),
        save_total_limit=2,
        metric_for_best_model="loss",  # f"eval_{source_name}_loss",
        load_best_model_at_end=True,
    )
    return training_args


def parse_args():
    parser = make_parser()
    parser.add_argument("--silver-labels", type=str, help="path to silver labels")
    return parser.parse_args()


def main():
    args = parse_args()

    processor = Wav2Vec2Processor.from_pretrained(
        args.processor_path
    )  # Run kaldi_dataset.py first to create the processor
    wer_metric = load_metric("wer")

    compute_metrics = make_metrics_calculator(wer_metric, processor)

    train_dataset, dev_dataset, test_dataset = make_dataset(
        args.source_name, args.target_name, args.silver_labels, processor
    )
    model = make_model(processor, args)
    data_collator = CTCCollator(processor)

    training_args = make_training_args(args.source_name, args.target_name, args)
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train(resume_from_checkpoint=args.resume is not None)

    metrics = trainer.evaluate(eval_dataset=test_dataset)

    print(f"Test metrics {metrics}")


if __name__ == "__main__":
    main()
