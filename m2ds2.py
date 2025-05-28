import math

import numpy as np
from datasets import (Dataset, concatenate_datasets, disable_caching)
from evaluate import load
from torch.utils.data import DataLoader
from transformers import (EarlyStoppingCallback, IntervalStrategy, Trainer,
                          TrainingArguments, Wav2Vec2Processor)
from transformers.trainer_utils import seed_worker

from collators import (CTCCollator, M2DS2Collator, MixedCollator,
                       PretrainCollator)
from models import Wav2Vec2ForCTCM2DS2
from samplers import DoubleSubsetRandomSampler, DoubleSubsetRandomSampler_Modified
from utils import (get_model,
                   make_metrics_calculator, make_parser)

import warnings
warnings.filterwarnings("ignore")

disable_caching()  # No need to consume disk space. data loading is pretty fast

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
        
        return dataloader


def make_dataset(args):
    def source_domain(batch):
        batch["domain"] = 0
        return batch

    def target_domain(batch):
        batch["domain"] = 1
        return batch

    def eval_domain(batch):
        batch["domain"] = -1
        return batch
    
    def remove_empty(batch):
        labels = batch['labels']
        return not (labels is None or (isinstance(labels, list) and len(labels) == 0) )

    dataset_source = Dataset.load_from_disk(f"{args.src_train_dir}")
    dataset_source = dataset_source.map(source_domain)
    dataset_target = Dataset.load_from_disk(f"{args.trg_train_dir}")
    print(f'LOG: Source data len: {len(dataset_source)}')

    dataset_target = dataset_target.shuffle()
    dataset_target = dataset_target.map(target_domain)
    print(f'LOG: Target data len: {len(dataset_target)}')
    dataset = concatenate_datasets([dataset_source,dataset_target ])  ## start from target

    print(f"LOG: Finetuning with mixed training on {args.src_name} source domain, {args.trg_name} target domain.")

    s_dataset_size = len(dataset_source)
    t_dataset_size = len(dataset_target)
    s_train_indices = list(range(s_dataset_size))
    t_train_indices = list(range(t_dataset_size))

    ## used to create batches that contain both domains
    train_sampler = DoubleSubsetRandomSampler(
        s_train_indices,
        t_train_indices,
        s_dataset_size,
        args.batch_size,
        args.batch_size*2,
    )
    
    print(dataset[0].keys())
    
    return dataset, train_sampler


def make_model(processor, args):
    model = get_model(
        processor=processor,
        model_cls=Wav2Vec2ForCTCM2DS2,
        params=args
    )
    print(f"SpecAugment {model.config.apply_spec_augment}")
    
    model.alpha = args.alpha
    model.beta = args.beta
    
    return model


def make_training_args(args):
    training_args = TrainingArguments(
        # output_dir="/content/gdrive/MyDrive/wav2vec2-large-xlsr-turkish-demo",
        logging_dir=f"./m2ds2-log/src_{args.src_name}_trg_{args.trg_name}",
        log_level='info',
        output_dir=f"{args.save_dir}/m2ds2-checkpoints/src_{args.src_name}_trg_{args.trg_name}",       ## save teachers 
        group_by_length=True,
        per_device_train_batch_size=args.batch_size,
        disable_tqdm=args.disable_tqdm,
        gradient_accumulation_steps=4,
        # evaluation_strategy="epoch",
        num_train_epochs=args.epochs,
        #max_steps=args.max_steps,
        fp16=True,
        save_strategy="epoch",
        #save_steps=args.eval_steps,
        #eval_steps=args.eval_steps,
        logging_strategy="epoch",
        #logging_steps=10,
        learning_rate=3e-4,
        warmup_steps=int(0.1 * args.max_steps),
        save_total_limit=20,    ## save last 10 epochs
        metric_for_best_model="loss",  # f"eval_{source_name}_loss",
    )
    return training_args


def parse_args():
    parser = make_parser(description="M2DS2 Training")
    parser.add_argument(
        '--disable-tqdm',
        action='store_true',
        help='Disable tqdm progrssion bar.'
    )
    parser.add_argument(
        '--src-train-dir',
        type=str,
        help='Path to the source train dataset directory.'
    )
    parser.add_argument(
        '--trg-train-dir',
        type=str,
        help='Path to the target train dataset directory.'
    )
    parser.add_argument(
        '--src-name',
        type=str,
        help='Name of source domain.'
    )
    parser.add_argument(
        '--trg-name',
        type=str,
        help='Name of target domain.'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.01,
        help='Alpha weight parameter'
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=0.01,
        help='Beta weight parameter'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        help='Path to save the model checkpoints.'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    processor = Wav2Vec2Processor.from_pretrained(
        args.processor_path
    )  # Run kaldi_dataset.py first to create the processor
    wer_metric = load("wer")

    compute_metrics = make_metrics_calculator(wer_metric, processor)

    train_dataset, train_sampler = make_dataset(args)
    
    model = make_model(processor, args)
    pre_collator = PretrainCollator(
        processor, model_config=model.config, include_domain=True
    )
    mixed_collator = MixedCollator(
        processor, model_config=model.config, include_domain=True
    )
    ctc_collator = CTCCollator(processor, include_domain=True)
    data_collator = M2DS2Collator(mixed_collator, ctc_collator, pre_collator)
    
    training_args = make_training_args(args)
    trainer = M2DS2Trainer(
        model=model,
        data_collator=mixed_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        tokenizer=processor.tokenizer,
        train_sampler=train_sampler,
    )

    trainer.train(resume_from_checkpoint=args.resume is not None)



if __name__ == "__main__":
    main()
