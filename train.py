from datasets import Dataset, disable_caching
from evaluate import load
from transformers import (EarlyStoppingCallback, IntervalStrategy, Trainer,
                          TrainingArguments, Wav2Vec2ForCTC, Wav2Vec2Processor)
from transformers.trainer_utils import seed_worker

from torch.utils.data import DataLoader

from models import Wav2Vec2ForCTCM2DS2

from collators import CTCCollator, M2DS2Collator
from utils import get_model, make_metrics_calculator, make_parser

import torch

import warnings
warnings.filterwarnings("ignore")

disable_caching()

def make_dataset(args):
    ### only ctc loss
    def source_domain(batch):
        batch["domain"] = torch.tensor(-1)
        return batch

    def eval_domain(batch):
        batch["domain"] = torch.tensor(-1)
        return batch
    
            
    train_dataset = Dataset.load_from_disk(f"{args.train_dir}")
    train_dataset = train_dataset.map(source_domain)
    
    
    print(f"LOG: Training on {args.dataset_name} source domain.")
    print(f'LOG: Dataset contains {len(train_dataset)} datapoints.')

    return train_dataset


def make_model(processor, args):
    model = get_model(
        processor=processor,
        model_cls=Wav2Vec2ForCTCM2DS2,
        params=args
    )
    return model


def make_training_args(args):
    training_args = TrainingArguments(
        logging_dir=f"./train-log/{args.dataset_name}",
        output_dir=f"{args.save_dir}/train-checkpoints/{args.dataset_name}",        ## save teachers 
        group_by_length=True,
        per_device_train_batch_size=args.batch_size,
        disable_tqdm=args.disable_tqdm,
        gradient_accumulation_steps=4,
        evaluation_strategy="no",
        num_train_epochs=args.epochs,
        # max_steps=1000,
        fp16=True,
        save_strategy="epoch",
        #save_steps=args.eval_steps,
        #eval_steps=args.eval_steps,
        logging_strategy="epoch",
        #logging_steps=10,
        learning_rate=3e-4,
        warmup_steps=int(0.1 * args.max_steps),
        save_total_limit=20,
        metric_for_best_model="loss",  # f"eval_{source_name}_loss",
    )
    return training_args


def parse_args():
    parser = make_parser(description="Supervised XLSR finetuning")
    parser.add_argument(
        '--disable-tqdm',
        action='store_true',
        help='Disable tqdm progression bar during training.'
    )
    parser.add_argument(
        '--train-dir',
        type=str,
        help='Path to the train dataset directory.'
    )
    parser.add_argument(
        '--dataset-name',
        type=str,
        help='Name of dataset for local checkpoints.'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        help='Path to save the model checkpoints.'
    )
    return parser.parse_args()

class M2DS2Trainer(Trainer):
    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        train_sampler=None,
    ):
        super(M2DS2Trainer, self).__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.train_sampler = train_sampler

    def _get_train_sampler(self):
        return self.train_sampler

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].
        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        train_sampler = self._get_train_sampler()

        dataloader = DataLoader(
                            train_dataset,
                            batch_size=self._train_batch_size,
                            sampler=train_sampler,
                            collate_fn=data_collator,
                            drop_last=self.args.dataloader_drop_last,
                            num_workers=self.args.dataloader_num_workers,
                            pin_memory=self.args.dataloader_pin_memory,
                            worker_init_fn=seed_worker,
                        )
        
        # print(next(iter(dataloader)))
        return dataloader

def main():
    args = parse_args()

    # Run kaldi_dataset.py first to create the processor
    processor = Wav2Vec2Processor.from_pretrained(
        args.processor_path
    )  
    print(processor)
    print(processor.tokenizer)
    print(processor.feature_extractor)
    wer_metric = load("wer")

    compute_metrics = make_metrics_calculator(wer_metric, processor)

    train_dataset = make_dataset(args)

    # print(train_dataset[0].keys())
    model = make_model(processor, args)
    ctc_collator = CTCCollator(processor,include_domain=True)
    #data_collator = M2DS2Collator(None, ctc_collator, None)
    print(f"LOG: Model loaded from {args.path_to_pretrained}")
    training_args = make_training_args(args)
    
    trainer = M2DS2Trainer(
        model=model,
        data_collator=ctc_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        tokenizer=processor.tokenizer,
    )
    print("LOG: Training started")
    train_results = trainer.train(resume_from_checkpoint=args.resume is not None)


if __name__ == "__main__":
    main()
