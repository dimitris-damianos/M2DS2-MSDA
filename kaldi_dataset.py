import argparse
import json
import os
import random
import re
import string
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Union

import datasets
import pandas as pd
import torch
import torchaudio
import torchaudio.functional as F
from transformers import (Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor,
                          Wav2Vec2Processor)

SR = 16_000


def parse_wav_scp(wav_scp: Union[str, Path]) -> Dict[str, str]:
    with open(wav_scp, "r") as fd:
        # Assume no pipes in wav.scp, i.e. lines in the format
        # uttid path/to/utterance.wav
        lines = [ln.strip().split() for ln in fd]

    return dict(lines)


def parse_text(text: Union[str, Path]) -> Dict[str, str]:
    with open(text, "r") as fd:
        text_lines = [ln.strip().split() for ln in fd]
    # Text in kaldi format
    # utterance_id1 sentence1
    # utterance_id2 sentence2
    transcriptions = {ln[0]: " ".join(ln[1:]) for ln in text_lines}
    return transcriptions


def strip_punctuation(sentence):
    punct = string.punctuation + "".join(["·", "»", "’", "‘", "´", "«"])
    return "".join(" " if i in punct else i for i in sentence.strip(punct))


def strip_accents(sentence):
    return "".join(
        c
        for c in unicodedata.normalize("NFD", sentence)
        if unicodedata.category(c) != "Mn"
    )


def normalize_text(sentence):
    sentence = sentence.strip()
    sentence = sentence.replace("<spoken_noise>", "[UNK]")
    sentence = sentence.replace("<spoken-noise>", "[UNK]")
    # sentence = strip_punctuation(sentence)
    sentence = strip_accents(sentence)
    _RE_COMBINE_WHITESPACE = re.compile(r"\s+")
    sentence = _RE_COMBINE_WHITESPACE.sub(" ", sentence).strip()
    sentence = sentence + " "
    return sentence


def kaldi_folder_manifest(
    kaldi_folder: str, domain: int
) -> Tuple[Dict[str, List], Dict[str, int]]:
    kaldi_path = Path(kaldi_folder)
    wavs = parse_wav_scp(kaldi_path / "wav.scp")
    transcriptions = parse_text(kaldi_path / "text")
    utterance_ids = list(transcriptions.keys())
    manifest = {
        "path": [wavs[utt_id] for utt_id in utterance_ids],
        "sentence": [
            normalize_text(transcriptions[utt_id]) for utt_id in utterance_ids
        ],
        "utt_id": utterance_ids,
        "domain": [domain for _ in utterance_ids],
    }
    vocab = [c for c in "αβγδεζηθικλμνξοπρςστυφχψωabcdefghijklmnopqrstuvwxyz "] + [
        "",
        "[UNK]",
        "[PAD]",
    ]

    vocab = {c: i for i, c in enumerate(vocab)}

    if not os.path.isfile("vocab.json"):
        with open("vocab.json", "w") as fd:
            json.dump(vocab, fd)

    import pprint

    print("VOCAB:")
    pprint.pprint(vocab)

    return manifest, vocab


def read_wav(batch, sr=16_000):
    wav, sr_org = torchaudio.load(batch["path"])
    if sr_org != sr:
        wav = F.resample(wav, sr_org, sr)
    wav = torch.mean(wav, dim=0)  # Downmix to mono
    batch["audio"] = wav
    return batch


def make_dataset(kaldi_folder: str, domain: int, sr: int = 16_000) -> datasets.Dataset:
    manifest, vocab = kaldi_folder_manifest(kaldi_folder, domain)
    dataset = datasets.Dataset.from_dict(manifest)
    # dataset = dataset.map(normalize_text)
    dataset = dataset.map(read_wav)  # , num_proc=16)

    def show_random_elements(dataset, num_examples=15):
        assert num_examples <= len(
            dataset
        ), "Can't pick more elements than there are in the dataset."
        picks = []
        for _ in range(num_examples):
            pick = random.randint(0, len(dataset) - 1)
            while pick in picks:
                pick = random.randint(0, len(dataset) - 1)
            picks.append(pick)

        df = pd.DataFrame(dataset[picks])
        print(df.to_string())

    show_random_elements(dataset)

    return dataset, vocab


def preprocess(
    dataset: datasets.Dataset, args, sr: int = 16_000
) -> Tuple[datasets.Dataset, Wav2Vec2Processor]:
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=sr,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    tokenizer = Wav2Vec2CTCTokenizer(
        "./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token=" "
    )
    if args.load_processor is not None:
        processor = Wav2Vec2Processor.from_pretrained(args.load_processor)
    else:
        processor = Wav2Vec2Processor(feature_extractor, tokenizer)

    print(processor)
    print(processor.tokenizer.get_vocab())
    
    def data_preparation(batch, sr=16_000):
        batch["input_values"] = processor(
            batch["audio"],
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        ).input_values[0]

        with processor.as_target_processor():
            batch["labels"] = processor(batch["sentence"]).input_ids
        return batch

    dataset = dataset.map(data_preparation, remove_columns=["audio"])
    return dataset, processor


def parse_args():
    parser = argparse.ArgumentParser("Save a kaldi folder as a huggingface dataset")
    parser.add_argument("--kaldi-folder", type=str, help="Load from kaldi folder")
    parser.add_argument(
        "--domain-id",
        type=int,
        help="Domain id - We used: {0: cv9, 1: hparl, 2: logotypografia}",
    )
    parser.add_argument(
        "--hf-folder", type=str, help="Save huggingface dataset to folder"
    )
    parser.add_argument(
        "--save-processor", type=str, default=None, help="Save processor to folder"
    )
    parser.add_argument(
        "--load-processor", type=str, default=None, help="Load processor from folder"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset, _ = make_dataset(args.kaldi_folder, args.domain_id, sr=SR)

    dataset, processor = preprocess(dataset, args, sr=SR)

    dataset.save_to_disk(args.hf_folder)
    if args.save_processor is not None:
        processor.save_pretrained(args.save_processor)


if __name__ == "__main__":
    main()
