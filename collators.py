from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    _compute_mask_indices, _sample_negative_indices)


def get_feat_extract_output_lengths(
    input_lengths: Union[torch.LongTensor, int], model_config
):
    """
    Computes the output length of the convolutional layers
    """

    def _conv_out_length(input_length, kernel_size, stride):
        # 1D convolutional layer output length formula taken
        # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        return (input_length - kernel_size) // stride + 1

    for kernel_size, stride in zip(model_config.conv_kernel, model_config.conv_stride):
        input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

    return input_lengths


def get_feature_vector_attention_mask(
    feature_vector_length: int,
    attention_mask: torch.LongTensor,
    model_config,
    add_adapter=None,
):
    # Effectively attention_mask.sum(-1), but not inplace to be able to run
    # on inference mode.
    non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

    output_lengths = get_feat_extract_output_lengths(non_padded_lengths, model_config)
    output_lengths = output_lengths.to(torch.long)

    batch_size = attention_mask.shape[0]

    attention_mask = torch.zeros(
        (batch_size, feature_vector_length),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    # these two operations makes sure that all values before the output lengths idxs are attended to
    attention_mask[
        (
            torch.arange(attention_mask.shape[0], device=attention_mask.device),
            output_lengths - 1,
        )
    ] = 1
    attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
    return attention_mask


def pad_input_values(
    features, processor, padding=True, max_length=None, pad_to_multiple_of=None
):
    input_features = [{"input_values": feature["input_values"]} for feature in features]
    batch = processor.pad(
        input_features,
        padding=padding,
        max_length=max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        return_attention_mask=True,
        return_tensors="pt",
    )

    return batch


def collate_for_ctc(
    partial_batch,
    uncollated_features,
    processor,
    padding=True,
    max_length_labels=None,
    pad_to_multiple_of_labels=None,
    include_domain=False,
    include_utt_id=False,
):
    """
    partial_batch: a partially collated batch. For example the batch returned from pad_input_values
    uncollated_features: dict of lists as it comes inside the collator
    """
    batch = partial_batch
    label_features = [
        {"input_ids": feature["labels"]} for feature in uncollated_features
    ]

    with processor.as_target_processor():
        labels_batch = processor.pad(
            label_features,
            padding=padding,
            max_length=max_length_labels,
            pad_to_multiple_of=pad_to_multiple_of_labels,
            return_tensors="pt",
        )

    # replace padding with -100 to ignore loss correctly
    labels = labels_batch["input_ids"].masked_fill(
        labels_batch.attention_mask.ne(1), -100
    )

    batch["labels"] = labels

    if include_domain and "domain" not in batch:
        domain = torch.tensor([feature["domain"] for feature in uncollated_features])
        batch["domain"] = domain

    if include_utt_id:
        # For decoding
        batch["utt_id"] = [f["utt_id"] for f in uncollated_features]
    # print(batch)
    return batch


def collate_for_pretrain(
    partial_batch, uncollated_features, model_config, include_domain=False
):
    """
    partial_batch: a partially collated batch. For example the batch returned from pad_input_values
    uncollated_features: dict of lists as it comes inside the collator
    """

    batch = partial_batch

    batch_size, raw_sequence_length = batch["input_values"].shape
    sequence_length = get_feat_extract_output_lengths(
        raw_sequence_length, model_config
    )  # .item()

    # make sure that no loss is computed on padded inputs
    if batch.get("attention_mask") is not None:
        # compute real output lengths according to convolution formula
        sub_attention_mask = get_feature_vector_attention_mask(
            sequence_length,
            batch["attention_mask"],
            model_config,
        )

    mask_time_indices = _compute_mask_indices(
        shape=(batch_size, sequence_length),
        mask_prob=model_config.mask_time_prob,
        mask_length=model_config.mask_time_length,
        attention_mask=sub_attention_mask,
    )
    
    sampled_negative_indices = _sample_negative_indices(
        features_shape=(batch_size, sequence_length),
        num_negatives=model_config.num_negatives,
        mask_time_indices=mask_time_indices,
    )
    mask_time_indices = torch.tensor(
        data=mask_time_indices,
        device=batch["input_values"].device,
        dtype=torch.long,
    )
    sampled_negative_indices = torch.tensor(
        data=sampled_negative_indices,
        device=batch["input_values"].device,
        dtype=torch.long,
    )

    batch["mask_time_indices"] = mask_time_indices
    batch["sampled_negative_indices"] = sampled_negative_indices

    if include_domain and "domain" not in batch:
        domain = torch.tensor([feature["domain"] for feature in uncollated_features])
        batch["domain"] = domain
    return batch


@dataclass
class CTCCollator:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    model_config: Any = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    include_domain: bool = False
    include_utt_id: bool = False

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        batch = pad_input_values(
            features,
            self.processor,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        batch = collate_for_ctc(
            batch,
            features,
            self.processor,
            padding=self.padding,
            max_length_labels=self.max_length_labels,
            pad_to_multiple_of_labels=self.pad_to_multiple_of_labels,
            include_domain=self.include_domain,
            include_utt_id=self.include_utt_id,
        )
        return batch


@dataclass
class PretrainCollator:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    model_config: Any = None
    include_domain: bool = False

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        batch = pad_input_values(
            features,
            self.processor,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        batch = collate_for_pretrain(
            batch, features, self.model_config, include_domain=self.include_domain
        )
        return batch


@dataclass
class MixedCollator:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    model_config: Any = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    model_config: Any = None
    include_domain: bool = False

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        batch = pad_input_values(
            features,
            self.processor,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        batch = collate_for_ctc(
            batch,
            features,
            self.processor,
            padding=self.padding,
            max_length_labels=self.max_length_labels,
            pad_to_multiple_of_labels=self.pad_to_multiple_of_labels,
            include_domain=self.include_domain,
        )

        batch = collate_for_pretrain(
            batch, features, self.model_config, include_domain=self.include_domain
        )
        return batch


class M2DS2Collator(object):
    def __init__(self, mixed_collator, ctc_collator, pre_collator):
        self.mixed_collator = mixed_collator
        self.pre_collator = pre_collator
        self.ctc_collator = ctc_collator

    def __call__(self, batch):
        domains = [b["domain"] for b in batch]
        # print(domains)
        if domains[0] < 0:  # we are in evaluation mode
            return self.ctc_collator(batch)
        if domains[0] == 0:  # source domain
            return self.mixed_collator(batch)
        if domains[0] == 1:  # target domain
            return self.pre_collator(batch)
