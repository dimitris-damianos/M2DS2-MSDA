import argparse

import numpy as np
from transformers import IntervalStrategy, TrainingArguments


def make_metrics_calculator(wer_metric, processor):
    def compute_metrics(pred):
        if pred is None or pred.predictions is None:
            pass
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        if any(len(s) == 0 for s in label_str):
            print(label_str)
            preds_fix, labels_fix = [], []
            for p, l in zip(pred_str, label_str):
                if len(l) > 0:
                    preds_fix.append(p)
                    labels_fix.append(l)
            if len(labels_fix) == 0:
                return {"wer", 0.0}
            label_str = labels_fix
            pred_str = preds_fix
        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    return compute_metrics


def make_parser(description="Experiment argument parser"):
    parser = argparse.ArgumentParser(description)
    parser.add_argument("--src-dataset", type=str, default="train5h",help="Dataset where domains belong")
    parser.add_argument("--trg-dataset", type=str, default="train5h",help="Dataset where domains belong")
    parser.add_argument("--source-domain", type=str, help="Source domain name")
    parser.add_argument("--target-domain", type=str, help="Target domain name")
    parser.add_argument("--processor-path", type=str, default="processor/", help="Path to saved processor")
    parser.add_argument("--apply-spec-aug",action='store_true',help="Whether or not to enable SpecAugment during finetuning.") 
    parser.add_argument("--apply-spec-aug-student",action='store_true',help="Whether or not to enable SpecAugment for student model.")  
    parser.add_argument("--path-to-pretrained",
                        type=str, 
                        default="facebook/wav2vec2-large-xlsr-53",
                        help="Pretrained model path. If local checkpoint, please set local-files-only=True") 
    parser.add_argument("--path-to-pretrained-student",
                        type=str, 
                        default="facebook/wav2vec2-large-xlsr-53",
                        help="Pretrained model path for studnet, used in META-PL. If local checkpoint, please set local-files-only-student=True")
    parser.add_argument("--pretrained-teacher",
                        type=str, 
                        default=None,
                        help="Path to meta pl pretrained teacher") 
    parser.add_argument("--local-files-only",
                        type=bool, 
                        default=False,
                        help="Load local files.") 
    parser.add_argument("--local-files-only-student",
                        type=bool, 
                        default=False,
                        help="Load local files for student pretrained model.")
    parser.add_argument("--meta-pl",type=bool,default=False,help="Used in META-PL training, for model loading and saving.")
    parser.add_argument(
        "--copt",
        action="store_true",
        help="Use domain adaptive pretraining. i.e. load dataset pretrained on target domain",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=20,
        help="num of training epochs",
    )
    parser.add_argument(
        "--log-step",
        type=int,
        default=1,
        help="log step",
    )
    parser.add_argument(
        "--save-step",
        type=int,
        default=1,
        help="save step",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="batch size")
    parser.add_argument(
        "--max-seconds", type=int, default=12, help="Use trimmed dataset to max_seconds"
    )
    parser.add_argument("--max-steps", type=int, default=10000, help="max_steps")
    parser.add_argument("--epochs", type=int, default=30, help="num of epochs")
    parser.add_argument("--world-size", type=int, default=1, help="Number of available GPUs")
    parser.add_argument(
        "--keep-percent", type=float, default=1.0, help="keep dataset percent"
    )
    parser.add_argument(
        "--eval-steps", type=int, default=1, help="evaluation step interval"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Checkpoint to resume training"
    )
    parser.add_argument(
        "--teacher-origin", type=str, default="FINETUNED", help="Teacher type: M2SD2 or FINETUNED"
    )
    parser.add_argument(
        "--train-type", type=str, default=None, help="Train type, e.g. feedback-ssl, feedback-ctc etc"
    )
    parser.add_argument(
        "--simple-meta-pl", type=bool, default=False, help="Whether or not to use simple (no aux task) meta pl training."
    )
    
    
    return parser


# def get_pretrained_model(args):
#     # Really ugly. Change according to your checkpoints
#     PRETRAINED_MODEL = {
#         "generic": "facebook/wav2vec2-large-xlsr-53",
#         "generic-small": "facebook/wav2vec2-base-960h",
#         "hparl": "xlsr-hparl/checkpoint-20000/",
#         "cv9": "xlsr-cv9/checkpoint-20000/",
#         "logotypografia": "xlsr-logotypografia/checkpoint-20000/",
#     }
#     if resume is None:
#         pretrained_model = (
#             PRETRAINED_MODEL[target_name] if copt else PRETRAINED_MODEL["generic"]
#         )

#         if copt:
#             print(f"loooog: loading {target_name} target domain dataset for copt")
#         else:
#             print(f"loooog: loading generic dataset without copt")
#     else:
#         pretrained_model = resume
#         print(f"Resuming training from {resume}")

#     return pretrained_model


# def load_model_from_pretrained(processor, model_cls, pretrained_model):
#     model = model_cls.from_pretrained(
#         # "facebook/wav2vec2-base-960h",
#         pretrained_model,
#         attention_dropout=0.1,
#         hidden_dropout=0.1,
#         feat_proj_dropout=0.0,
#         mask_time_prob=0.4,
#         layerdrop=0.1,
#         ctc_loss_reduction="mean",
#         pad_token_id=processor.tokenizer.pad_token_id,
#         vocab_size=len(processor.tokenizer),
#     )
#     # model = model.cuda()
#     model.config.ctc_zero_infinity = True

#     model.freeze_feature_extractor()
#     model.gradient_checkpointing_enable()
#     return model



## my custom "get model" that is controled via parser arguments
def get_model(processor,model_cls,params,model_type=None):
    if params.meta_pl == False:
        #print(f'Pretrained path: {params.path_to_pretrained}')
        model = model_cls.from_pretrained(   ### to resume training, simply provide the last checkpoint
            params.path_to_pretrained,
            apply_spec_augment = params.apply_spec_aug,
            local_files_only=params.local_files_only,
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
        
    elif params.meta_pl == True:
        if model_type=="student":
            #print(f'Pretrained path: {params.path_to_pretrained_student} (meta-pl)')
            model = model_cls.from_pretrained(   ### to resume training, simply provide the last checkpoint
                params.path_to_pretrained_student,
                apply_spec_augment = params.apply_spec_aug_student,
                local_files_only=params.local_files_only_student,
                ## hardcoded params
                attention_dropout=0.1,
                hidden_dropout=0.1,
                feat_proj_dropout=0.0,
                mask_time_prob=0.5,
                mask_time_length=5, ## default is 10, caused problems with masked indices
                mask_time_min_masks=5,
                layerdrop=0.1,
                ctc_loss_reduction="mean",
                pad_token_id=processor.tokenizer.pad_token_id,
                vocab_size=len(processor.tokenizer),
            )
            model.config.ctc_zero_infinity = True

            model.freeze_feature_extractor()
            model.gradient_checkpointing_enable()
        else:
            #print(f'Pretrained path: {params.path_to_pretrained} (meta-pl)')
            model = model_cls.from_pretrained(   ### to resume training, simply provide the last checkpoint
                params.path_to_pretrained,
                apply_spec_augment = params.apply_spec_aug,
                local_files_only=params.local_files_only,
                ## hardcoded params
                attention_dropout=0.1,
                hidden_dropout=0.1,
                feat_proj_dropout=0.0,
                mask_time_prob=0.5,
                mask_time_length=5, ## default is 10, caused problems with masked indices
                mask_time_min_masks=5,
                layerdrop=0.1,
                ctc_loss_reduction="mean",
                pad_token_id=processor.tokenizer.pad_token_id,
                vocab_size=len(processor.tokenizer),
            )
            model.config.ctc_zero_infinity = True

            model.freeze_feature_extractor()
            model.gradient_checkpointing_enable()
    #print("Spec augment:",model.config.apply_spec_augment)
    return model
