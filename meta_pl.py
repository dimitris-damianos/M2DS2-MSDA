import math
import os

import numpy as np
from datasets import (Dataset, concatenate_datasets, disable_caching)

from torch.utils.data import DataLoader

from torch import distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group,  destroy_process_group
import os

from transformers import (EarlyStoppingCallback, IntervalStrategy, Trainer,
                          TrainingArguments, Wav2Vec2Processor)
from transformers.trainer_utils import seed_worker

from collators import (CTCCollator, M2DS2Collator, MixedCollator,
                       PretrainCollator)
from models import Wav2Vec2ForCTCM2DS2
from samplers import DoubleSubsetRandomSampler, DistributedDoubleSubsetRandomSampler

from utils import (get_model, make_metrics_calculator, make_parser)

from torch.optim.adamw import AdamW

import warnings
warnings.filterwarnings("ignore")

import torch
import logging
import sys

from tqdm import tqdm 
import socket

### Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)  
formatter = logging.Formatter('%(asctime)s  - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def parse_args():
    parser = make_parser(description="META PL Training")
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
        '--valid-dir',
        type=str,
        help='Path to the validation dataset directory.'
    )
    parser.add_argument(
        '--test-dir',
        type=str,
        help='Path to the test dataset directory.'
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
        '--student-path',
        type=str,
        default=None,
        help="Student checkpoint location (local/HF)."
    )
    parser.add_argument(
        '--use-local-student',
        action="store_true",
        help="Enable if you want to use local student checkpoints."
    )
    parser.add_argument(
        '--student-spec-augment',
        action="store_true",
        help="Apply SpecAugment to student's input."
    )
    parser.add_argument(
        '--teacher-path',
        type=str,
        required=True,
        help="Local teacher checkpoint location."
    )
    parser.add_argument(
        '--teacher-spec-augment',
        action="store_true",
        help="Apply SpecAugment to teacher's input."
    )
    parser.add_argument(
        '--disable-tqdm',
        action='store_true',
        help='Disable tqdm progression bar.'
    )
    parser.add_argument(
        '--exp-save-path',
        type=str,
        help='Path to experiments save directory (where Teacher and Student checkpoints are saved).'
    ) 
    return parser.parse_args()

def get_pseudolabels(processor,model=None,batch=None):
    model.training = False  ## teacher is not training
    with torch.no_grad():
        out = model(input_values=batch["input_values"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    domain=batch["domain"])
    model.training = True    

    logits = out.logits
    pred_ids = torch.argmax(logits, axis=-1)
    ### decode logits
    pred_str = processor.batch_decode(pred_ids)
    #### If empty, replace with ' ' to avoid labels batch being empty
    pred_str = [sentence if sentence.strip() != "" else ' ' for sentence in pred_str]
    
    batch["sentence"] = pred_str
    with processor.as_target_processor():
        ### encode pseudo-target
        features = processor(batch["sentence"]).input_ids
    
    ### pad pseudo-targets
    label_features = [
        {"input_ids": feature} for feature in features
    ]
    with processor.as_target_processor():
        labels_batch = processor.pad(
            label_features,
            padding=True,
            max_length=True,
            pad_to_multiple_of=True,
            return_tensors="pt",
        )
    labels = labels_batch["input_ids"].masked_fill(
        labels_batch.attention_mask.ne(1), -100
    )
    ### create batch with pseudo-targets
    batch["labels"] = labels
    return batch

## create datasets
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
    dataset_source = dataset_source.shuffle()
    dataset_source = dataset_source.map(source_domain)
    print(f'LOG: Source data len: {len(dataset_source)}')

    dataset_target = Dataset.load_from_disk(f"{args.trg_train_dir}")
    dataset_target = dataset_target.shuffle()
    dataset_target = dataset_target.map(target_domain)
    print(f'LOG: Target data len: {len(dataset_target)}')
    dataset = concatenate_datasets([dataset_source,dataset_target])  ## start from target

    source_dev_dataset = Dataset.load_from_disk(f"{args.valid_dir}")
    source_dev_dataset = source_dev_dataset.map(source_domain)
    dev_dataset = source_dev_dataset
    
    test_dataset = Dataset.load_from_disk(f"{args.test_dir}")
    test_dataset = test_dataset.map(eval_domain)
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
    return dataset, dev_dataset, test_dataset, train_sampler

def make_dataloader(train_sampler,collator_fn, dataset,args):
    return DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            collate_fn=collator_fn,
            worker_init_fn=seed_worker,
            
        )

def make_teacher(processor,args):
    logger.info(f'Teacher loaded from {args.teacher_path}.')
    model = Wav2Vec2ForCTCM2DS2.from_pretrained(
        args.teacher_path,
        apply_spec_augment = args.teacher_spec_augment,
        local_files_only=True,
        ## hardcoded params
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.5,
        mask_time_length=5, ## default is 10, caused problems with masked indices
        mask_time_min_masks=2,
        layerdrop=0.1,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )
    model.config.ctc_zero_infinity = True
    model.freeze_feature_extractor()
    model.gradient_checkpointing_enable()
    return model


def make_student(processor,args):
    logger.info(f'Student loaded from {args.student_path}.')
    model = Wav2Vec2ForCTCM2DS2.from_pretrained(
        args.student_path,
        apply_spec_augment = args.student_spec_augment,
        local_files_only=args.use_local_student,
        ## hardcoded params
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.5,
        mask_time_length=5, ## default is 10, caused problems with masked indices
        mask_time_min_masks=2,
        layerdrop=0.1,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )
    model.config.ctc_zero_infinity = True
    model.freeze_feature_extractor()
    model.gradient_checkpointing_enable()
    return model


def student_train_step(
        student, teacher, 
        optimizer, batch, processor
    ):
    """
    Train student model on target domain (domain = 1) using teacher's PLs.
    """
    ## Get PLs from target domain
    target_batch = get_pseudolabels(processor=processor,
                                    model=teacher,
                                    batch=batch)
    
    ## simple ctc forward for student
    ctc_loss = student.ctc_forward(input_values=target_batch["input_values"],
                   attention_mask=target_batch["attention_mask"],
                   labels=target_batch["labels"]).loss
    
    loss = ctc_loss 
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss

def teacher_train_step(
        student, 
        optimizer, batch,
    ):
    """
    Train teacher on student's feedback on the source domain.
    """
    loss = student.ctc_forward(  ### ctc-forward
                input_values=batch["input_values"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]).loss
        
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss


def meta_pl_step(student, teacher, student_optim, teacher_optim, batch, processor, args):
    """
    Perform a single training step of the META PL framework.
    """
    teacher_loss, student_loss = 0, 0
    if batch["domain"][0]==1: ## target domain batch
        ## Only teacher's PL on target for student
        student_loss = student_train_step(student=student,
                           teacher=teacher,
                           optimizer=student_optim,
                           batch=batch,
                           processor=processor)
        
    elif batch["domain"][0]==0: ## source domain batch
        ## Only student's feedback for teacher
        teacher_loss = teacher_train_step(student=student,
                           optimizer=teacher_optim,
                           batch=batch,
                           )
        
    return teacher_loss, student_loss
    
def save_checkpoint(location, model, args):
    os.makedirs(location,exist_ok=True)
    torch.save(model.state_dict(), os.path.join(location,f"checkpoint-{args.save_incr}.pt"))


def run(args):

    processor = Wav2Vec2Processor.from_pretrained(
        args.processor_path
    )  # Run kaldi_dataset.py first to create the processor
    
    train_dataset, test_dataset, dev_dataset, train_sampler = make_dataset(args)
    
    student = make_student(processor, args)
    teacher = make_teacher(processor, args)
    
    mixed_collator = MixedCollator(
        processor, model_config=teacher.config, include_domain=True
    )
    
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    logger.info(f"Device: {device}")

    ### we apply ctc loss on both domains, and pre-training as well
    train_loader = make_dataloader(train_sampler,mixed_collator,train_dataset,args)
    
    student_optim = AdamW(params=student.parameters(),lr=3e-4)
    teacher_optim = AdamW(params=teacher.parameters(),lr=3e-4)

    args.save_incr = 0
    args.iter_counter = 0

    student_checkpoint = f"{args.exp_save_path}/{args.teacher_origin}/src_{args.src_name}_trg_{args.trg_name}/student/"
    teacher_checkpoint = f"{args.exp_save_path}/{args.teacher_origin}/src_{args.src_name}_trg_{args.trg_name}/teacher/"
    os.makedirs(student_checkpoint,exist_ok=True)
    os.makedirs(teacher_checkpoint,exist_ok=True)

    student.to(device)
    teacher.to(device)
    
    teacher.train()
    student.train()
    
    logger.info("Training started")
    for epoch in range(args.num_epochs):
        iter_per_epoch = 0
        epoch_t_loss, epoch_s_loss = 0, 0
        num_batches = len(train_loader)
        
        if not bool(args.disable_tqdm):
            pbar = tqdm(enumerate(train_loader), total=num_batches, desc=f'Epoch {epoch}/{args.num_epochs}')
        else:
            pbar = enumerate(train_loader)
        for idx, batch in pbar:
            iter_per_epoch +=1
            batch = batch.to(device)
            t_loss, s_loss = meta_pl_step(
                student=student, teacher=teacher,
                student_optim=student_optim, teacher_optim=teacher_optim,
                batch=batch, processor=processor, args=args
            )
            epoch_t_loss += t_loss
            epoch_s_loss += s_loss

        if args.save_incr % args.log_step == 0:
            logger.info(f"Epoch {epoch}/{args.num_epochs}, teacher loss: {epoch_t_loss/iter_per_epoch}, student loss: {epoch_s_loss/iter_per_epoch}")
        if args.save_incr % args.save_step == 0:
            save_checkpoint(location=student_checkpoint,model=student,args=args)
            save_checkpoint(location=teacher_checkpoint,model=teacher,args=args)
            args.save_incr += 1
        args.iter_counter += 1

def main():
    args = parse_args()
    run(args=args)
    

if __name__=="__main__":
   main()
