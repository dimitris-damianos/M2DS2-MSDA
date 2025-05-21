import argparse
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import Dataset
from evaluate import load
from pyctcdecode import build_ctcdecoder
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (Wav2Vec2ForCTC, Wav2Vec2Processor,
                          Wav2Vec2ProcessorWithLM)
from models import Wav2Vec2ForCTCM2DS2
from utils import (get_model, make_metrics_calculator, make_parser)

import os

import warnings
warnings.filterwarnings("ignore")

SAVE_PATH = "/nfs1/ddamianos/msda"

def parse_args():
    parser = argparse.ArgumentParser("Decoding")
    parser.add_argument(
        '--model-checkpoint',
        type=str,
        help='Path to the model checkpoint'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        help='Model name to use in prediction results'
    )
    parser.add_argument('--dataset', type=str, help='Dataset to use')
    parser.add_argument("--dataset-name", type=str, help="dataset name to decode")
    
    parser.add_argument(
        "--predictions-folder",
        type=str,
        default=None,
        help="Write predictions to folder",
    )
    parser.add_argument(
        "--cached-logits",
        type=str,
        default=None,
        help="Used cached logits",
    )
    parser.add_argument("--lm", type=str, default=None, help="path to language model")
    parser.add_argument(
        "--unigrams", type=str, default=None, help="path to list of unigrams"
    )
    parser.add_argument(
        "--extract-silver-labels",
        action="store_true",
        help="Extract labels for pseudolabeling",
    )
    parser.add_argument(
        "--dump-hidden-states", action="store_true", help="save the hidden states"
    )

    parser.add_argument(
        "--dump-codevectors", action="store_true", help="save the codevectors"
    )
    parser.add_argument(
        "--dump-logits", action="store_true", help="save the hidden states"
    )
    parser.add_argument(
        "--dump-conv",
        action="store_true",
        help="dump features xtracted by the conv network",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Use dev set. If False uses test set",
    )
    parser.add_argument(
        '--processor',
        type=str,
        default='./processor/',
        help='Pretrained processor location'
    )
    parser.add_argument('--is-metapl',action='store_true',help='Used when loading local meta pl checkpoint')
    parser.add_argument("--batch-size", type=int, default=8, help="batch size")
    return parser.parse_args()


def read_unigrams(uni_path):
    with open(uni_path, "r") as fd:
        unigrams = [ln.strip() for ln in fd]
    return unigrams


def get_processor(args):
    processor = Wav2Vec2Processor.from_pretrained(args.processor) ### FIX ME
    # if args.lm is not None:
    #     vocab_dict = processor.tokenizer.get_vocab()
    #     # vocab_dict["<unk>"] = vocab_dict.pop("[UNK]")
    #     # vocab_dict["<pad>"] = vocab_dict.pop("[PAD]")
    #     sorted_vocab_dict = {
    #         k.lower(): v
    #         for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])
    #     }
    #     decoder = build_ctcdecoder(
    #         labels=list(sorted_vocab_dict.keys()),
    #         kenlm_model_path=args.lm,
    #         unigrams=read_unigrams(args.unigrams)
    #         if args.unigrams is not None
    #         else None,
    #     )

    #     processor = Wav2Vec2ProcessorWithLM(
    #         processor.feature_extractor, processor.tokenizer, decoder
    #     )

    return processor


ARGS = parse_args()
processor = get_processor(ARGS)
wer_metric = load("wer")

# Copied from Accelerate.
def _pad_across_processes(tensors, pad_index=-100):
    """
    Recursively pad the tensors in a nested list/tuple/dictionary of tensors from all devices to the same size so
    they can safely be gathered.
    """
    max_size = max([l.shape[1] for l in tensors])

    # Then pad to the maximum size
    old_sizes = [t.shape for t in tensors]
    new_size = [list(old_size) for old_size in old_sizes]
    for i in range(len(new_size)):
        new_size[i][1] = max_size
    new_tensors = [
        tensor.new_zeros(tuple(new_sz)) + pad_index
        for tensor, new_sz in zip(tensors, new_size)
    ]
    for i in range(len(new_tensors)):

        new_tensors[i][:, : old_sizes[i][1]] = tensors[i]
    return torch.cat(new_tensors, dim=0)


@dataclass
class DataCollatorWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    model_config: Any = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        batch["utt_id"] = [
            f["utt_id"] if 'utt_id' in f.keys() else f['client_id'] for f in features
            ]
        return batch


def make_dataset(args):
    dataset = Dataset.load_from_disk(f"{args.dataset}")
    print(f"LOG: Using {args.dataset_name}, from {args.dataset}")
    return dataset



def make_model(processor,args):
    print(f'Model loaded from {args.model_checkpoint}.')
    if args.is_metapl:
        checkpoint = torch.load(args.model_checkpoint)
        model = Wav2Vec2ForCTCM2DS2.from_pretrained(
            'facebook/wav2vec2-large-xlsr-53',
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
        model.load_state_dict(checkpoint)
    else:
        model = Wav2Vec2ForCTCM2DS2.from_pretrained(
            args.model_checkpoint,
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
    
    model = model.cuda()
    model.eval()
    return model


def decode_predictions_no_lm(pred_logits):
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred_str = processor.batch_decode(pred_ids)
    return pred_str


def decode_predictions(pred_logits):
    pred_str = [
        processor.decode(lg.numpy(),beam_width=13).text.replace("⁇", "") for lg in tqdm(pred_logits)
    ]
    return pred_str


def compute_metrics(pred_str, label_ids):
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    label_str = processor.tokenizer.batch_decode(label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def write_silver_labels(out_file, pred_str, speaker_ids):
    with open(out_file, "w") as fd:
        for utt_id, sent in zip(speaker_ids, pred_str):
            fd.write(f"{sent} ( {utt_id} )\n")


def evaluation_loop(model, dataloader, args):
    model.eval()
    all_logits = []
    all_labels = []
    all_speaker_ids = []
    all_hidden_states = []
    with torch.no_grad():
        print("Evaluating")
        for batch in tqdm(dataloader):
            speaker_ids = batch.pop("utt_id")
            batch = {
                k: v.cuda()
                for k, v in batch.items()
            }
            # print(batch)
            out = model(**batch, output_hidden_states=True)
            hidden_states = out.hidden_states[-1].detach().cpu()
            all_logits.append(out.logits.detach().cpu())
            all_labels.append(batch["labels"].detach().cpu())
            all_speaker_ids += speaker_ids
            if args.dump_hidden_states:
                all_hidden_states += [
                    hidden_states[i, ...] for i in range(hidden_states.size(0))
                ]
    return all_speaker_ids, all_logits, all_labels, all_hidden_states


def flatten_list_of_tensors(tensors):
    return [t for batch in tensors for t in batch]


def decode(logits, args):
    decode_fn = decode_predictions_no_lm
    # decode_fn = decode_predictions if args.lm is not None else decode_predictions_no_lm
    logits = (
        flatten_list_of_tensors(logits)
        if args.lm is not None
        else _pad_across_processes(logits)
    )

    predicted_str = decode_fn(logits)
    return predicted_str


def dump_hidden_states(speaker_ids, hidden_states, args):
    hidden_states_path = f"{args.predictions_folder}/{args.file_prefix}_states.p"

    if args.dump_hidden_states:
        with open(hidden_states_path, "wb") as fd:
            dat = dict(zip(speaker_ids, hidden_states))
            pickle.dump(dat, fd)


def dump_logits(speaker_ids, logits, labels, args):
    logits_path = f"{args.predictions_folder}/{args.file_prefix}_logits.p"

    if args.dump_logits:
        with open(logits_path, "wb") as fd:
            pickle.dump((speaker_ids, logits, labels), fd)


def dump_codevectors(codevectors, projected_codevectors, args):
    codevectors_path = f"{SAVE_PATH}/codevectors/{args.model_name}_{args.dataset_name}_codevectors.pt"
    projected_codevectors_path = (
        f"{SAVE_PATH}/projected_codevectors/{args.model_name}_{args.dataset_name}_projected_codevectors.pt"
    )

    if args.dump_codevectors:
        torch.save(codevectors,codevectors_path)
        torch.save(projected_codevectors, projected_codevectors_path)


def dump_text(speaker_ids, predicted_str, args):
    pred_str_path = f"{args.predictions_folder}/{args.model_name}/{args.dataset_name}_text.txt"

    write_silver_labels(pred_str_path, predicted_str, speaker_ids)


def make_silver_labels(speaker_ids, logits, args):
    predicted_str = decode(logits, args)

    dump_text(speaker_ids, predicted_str, args)


def evaluate(speaker_ids, logits, labels, args):
    predicted_str = decode(logits, args)

    labels = _pad_across_processes(labels, -100)
    wer = compute_metrics(predicted_str, labels)
    
    dump_text(speaker_ids, predicted_str, args)

    return wer


def get_codevectors(model, dataloader):
    model.eval()
    all_projected_codevectors = []
    all_codevectors = []
    with torch.no_grad():
        print("Extracting codevectors")
        for batch in tqdm(dataloader):
            # speaker_ids = batch.pop("utt_id")
            batch = {
                'input_values': batch['input_values'].cuda(),
            }
            out = model.get_code_vectors(**batch, output_hidden_states=True)
            codevectors = out.hidden_states.detach().cpu()  # ugly  hack
            projected_codevectors = out.logits.detach().cpu()
            all_codevectors += [codevectors[i, ...] for i in range(codevectors.size(0))]

            all_projected_codevectors += [
                projected_codevectors[i, ...] for i in range(codevectors.size(0))
            ]
    return all_codevectors, all_projected_codevectors

def get_conv_features(model, dataloader):
    model.eval()
    all_features = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_values = batch["input_values"].cuda()
            out = model.wav2vec2.feature_extractor(input_values)
            features = out.detach().cpu()  # ugly  hack
            all_features += [features[i, ...] for i in range(features.size(0))]

    return all_features


def dump_conv_features(features, args):
    features_path = f"{args.predictions_folder}/{args.file_prefix}_conv_features.p"
    if args.dump_conv:
        with open(features_path, "wb") as fd:
            pickle.dump(features, fd)

def dump_reference(dataloader,args):
    ref_str_path = f"{args.predictions_folder}/{args.model_name}/{args.dataset_name}_ref_text.txt"
    os.makedirs(os.path.dirname(f"{args.predictions_folder}/{args.model_name}/"),exist_ok=True)
    with open(ref_str_path,"w") as f:
        print("Dumping reference transcriptions")
        for batch in tqdm(dataloader):
            label = batch['labels'][0]
            dec_labels = [
            processor.decode(lg.numpy()).replace("⁇", "") for lg in label
                ]
            transcipt = "".join([chr if chr !="" else " " for chr in dec_labels])
            f.write(f"{transcipt} \n")

def main():
    args = parse_args()
    
    ## check if destination folder for predictions exists
    # os.makedirs(args.predictions_folder,exist_ok=True)
    os.makedirs(f"{args.predictions_folder}/{args.model_name}/",exist_ok=True)
    
    if args.cached_logits is not None and os.path.isfile(args.cached_logits):
        with open(args.cached_logits, "rb") as fd:
            speaker_ids, logits, labels = pickle.load(fd)
    else:
        dataset = make_dataset(args )
        # dataset = dataset.select(range(10))
        model = make_model(processor, args)
        
        ### FIXME If utt_id is required in results, fixe collator to maintain field in batch
        data_collator = DataCollatorWithPadding(processor, model_config=model.config)

        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, collate_fn=data_collator
        )
        speaker_ids, logits, labels, hidden_states = evaluation_loop(
                model, dataloader, args
            )
        # dump_reference(dataloader=dataloader,args=args)
        if args.dump_codevectors:
            codevectors, projected_codevectors = get_codevectors(model, dataloader)
            dump_codevectors(codevectors, projected_codevectors, args)
            return
        elif args.dump_conv:
            conv_features = get_conv_features(model, dataloader)
            dump_conv_features(conv_features, args)
        else:
            # speaker_ids, logits, labels, hidden_states = evaluation_loop(
            #     model, dataloader, args
            # )

            if args.dump_logits:
                dump_logits(speaker_ids, logits, labels, args)

            if args.dump_hidden_states:
                dump_hidden_states(speaker_ids, hidden_states, args)

    if args.extract_silver_labels:
        make_silver_labels(speaker_ids, logits, args)
    else:
        # speaker_ids, logits, labels, hidden_states = evaluation_loop(
        #         model, dataloader, args
        #     )
        metrics = evaluate(speaker_ids, logits, labels, args)
        print(metrics)
        with open(f"{args.predictions_folder}/{args.model_name}/{args.dataset_name}_metrics.csv", "a") as fd:
            lm_ = args.lm if args.lm is not None else "N/A"
            fd.write(
                    f"{args.model_name}\t{args.model_checkpoint}\t{lm_}\t{metrics['wer']}\n"
                )


if __name__ == "__main__":
    main()
