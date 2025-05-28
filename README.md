# MSDA: Combining Pseudo-labeling and Self-Supervision for Unsupervised Domain Adaptation in ASR
This repository is the official implementation of the INTERSPEECH 2025 paper:

Below we provide all the required steps for enviroment setup, data preperation and training implementations for stage 1 and 2 of our method. 
## Enviroment setup
Requires `anaconda`/`miniconda` installed.
Run the following to create a conda enviroment with all the required dependencies: 
```bash
conda create -n msda python=3.10
conda activate msda
pip install -r requirements.txt
```

## Data preperation
To create a preprocessed and normalized Huggingface dataset from kaldi format, run:
```bash
python kaldi_dataset.py \
    --kaldi-folder /path/to/kaldi/dir \
    --hf-folder /hf/save/location \
    --load-procesor /processor \        # load existing processor     
    --save-processor ./processor        # create new processor
```
The desired dataset format is:
```bash
Dataset({
    features: ['file', 'text', 'speaker_id', 'chapter_id', 'id', 'input_values', 'labels'],
    num_rows: XXX
})
```
This format is required in all training and evaluation steps.
## Finetuning
To simply finetune XLSr-53 to the desired dataset, run:
```bash
python ./train.py \
    --path-to-pretrained facebook/wav2vec2-large-xlsr-53\
    --train-dir /path/to/dataset \                
    --dataset-name /dataset/name \          # Used for results naming  
    --epochs 30 \
    --apply-spec-aug \                      # Enable to apply SpecAugment
    --save-dir \path\to\save                # Checkpoint save location
    --disable-tqdm                          # Enable for tqdm
```

## Stage 1
In this stage, we apply the methodology described in M2DS2 to create a teacher model, using:
```bash
python ./m2ds2.py \
      --path-to-pretrained facebook/wav2vec2-large-xlsr-53 \
      --src-train-dir /path/to/source \
      --trg-train-dir /path/to/target \
      --src-name  /source/name \            # Used for results naming
      --trg-name  /target/name \            # Used for results naming
      --batch-size 4 \
      --epochs 30 \
      --apply-spec-aug \                    # Enable to apply SpecAugment
      --save-dir \path\to\save              # Checkpoint save location
      --disable-tqdm                        # Enable for tqdm
```

## Stage 2
For the stage 2, choose between the 2 following approaches to train a student model.
### Meta PL
We apply the methodology described in Meta Pseudo Labels to create a student model, using:
```bash
python ./meta_pl.py \
    --src-train-dir path/to/source/dataset/ \
    --trg-train-dir path/to/target/dataset/ \
    --src-name  /source/name \                      # Used for results naming
    --trg-name  /target/name \                      # Used for results naming
    --batch-size 4 \
    --student-path facebook/wav2vec2-large-xlsr-53 \    
    --student-spec-augment \                        # Enable to apply SpecAugment to student training 
    --teacher-path /path/to/pretrained/teacher \
    --log-step 1 \
    --save-step 1 \
    --teacher-origin /type/of/teacher \             #Type of teacher: M2DS2 trained of Finetuned
    --num-epochs 30 \
    --exp-save-path /path/to/save \
    --disable-tqdm
```
### MSDA
We apply the methodology described in our paper to create a student model, using:
```bash
python ./msda.py \
    --src-train-dir path/to/source/dataset/ \
    --trg-train-dir path/to/target/dataset/ \
    --src-name  /source/name \                      # Used for results naming
    --trg-name  /target/name \                      # Used for results naming
    --batch-size 4 \
    --student-path facebook/wav2vec2-large-xlsr-53 \     
    --student-spec-augment \                        # Enable to apply SpecAugment to student training 
    --teacher-path /path/to/pretrained/teacher \
    --log-step 1 \
    --save-step 1 \
    --teacher-origin /type/of/teacher \             # Type of teacher: M2DS2 trained of Finetuned
    --num-epochs 30 \
    --delta 2e-5 \
    --gamma 1e-5 \
    --keep-percent 0.X \                            # Target data percentage to keep for adaptation  
    --exp-save-path /path/to/save \
    --disable-tqdm
```
Keep in mind that both `msda.py` and `meta.pl` save both student and teacher models during training.
## Evaluation
To evaluate all trained models (finetuned, M2DS2, Meta PL, MSDA), run:
```bash
python ./decode.py \
    --model-checkpoint  /path/to/model \
    --model-name /model/name \
    --dataset /path/to/dataset \            # Used for results naming
    --dataset-name /dataset/name \          # Used for results naming
    --predictions-folder ./predictions \
    --batch-size 4 \
    --is-metapl                             # Enable for Meta PL/MSDA local checkpoints
```
## Citation

## Contact
- d.damianos@athena.rc