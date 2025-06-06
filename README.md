# Self & Semi-supervised Speech Domain Adaptation (MSDA, M2DS2)
This repository contains the official implementations of the following 2 papers: 

- [Sample-Efficient Unsupervised Domain Adaptation of Speech Recognition Systems A case study for Modern Greek](https://arxiv.org/abs/2301.00304)

- [MSDA: Combining Pseudo-labeling and Self-Supervision for Unsupervised Domain Adaptation in ASR](https://arxiv.org/abs/2505.24656)

We provide all necessary steps for setting up the environment, preparing data, and running training pipelines for both Mixed Multi-Domain
Self-Supervision (M2DS2) and Multi Stage Domain Adaptation (MSDA). 
## Enviroment setup
Both implementations share the same Python environment.
Ensure you have `conda` (Anaconda or Miniconda) installed, then run:
```bash
conda create -n exp-env python=3.10
conda activate exp-env
pip install -r requirements.txt
```

## Data preperation
Both approaches rely on a common dataset format.
To convert a dataset from Kaldi format to a preprocessed and normalized Hugging Face `datasets` format, run:
```bash
python kaldi_dataset.py \
    --kaldi-folder /path/to/kaldi/dir \
    --hf-folder /hf/save/location \
    --load-procesor /processor \        # Optional: Load existing processor     
    --save-processor ./processor        # Optional: Save new processor
```
Expected dataset format:
```bash
Dataset({
    features: ['file', 'text', 'speaker_id', 'chapter_id', 'id', 'input_values', 'labels'],
    num_rows: XXX
})
```
⚠️ This format is required for all training and evaluation steps
## Finetuning
To fine-tune XLS-R on your dataset:
```bash
python train.py \
    --path-to-pretrained facebook/wav2vec2-large-xlsr-53 \
    --train-dir /path/to/dataset \
    --dataset-name dataset_name \
    --epochs 30 \
    --apply-spec-aug \                      # Optional: Enable SpecAugment
    --save-dir /path/to/save \
    --disable-tqdm                          # Optional: Disable tqdm progress bar
```

## M2DS2: Mixed Multi-Domain Self-Supervision
To train a model using the M2DS2 approach, run:
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

## MSDA: Multi Stage Domain Adaptation
MSDA leverages both fine-tuned and M2DS2-trained models as teachers to guide the training of a student model for improved domain adaptation.

You can choose between two student training strategies:
### Meta Pseudo Labels (Meta PL)
This approach applies the [Meta Pseudo Labels](https://arxiv.org/abs/2003.10580) method to train a student model with teacher guidance:
```bash
python ./meta_pl.py \
    --src-train-dir path/to/source/dataset/ \
    --trg-train-dir path/to/target/dataset/ \
    --src-name  /source/name \                      # For logging and results
    --trg-name  /target/name \                      # For logging and results
    --batch-size 4 \
    --student-path facebook/wav2vec2-large-xlsr-53 \    
    --student-spec-augment \                        # Enable to apply SpecAugment to student training 
    --teacher-path /path/to/pretrained/teacher \
    --log-step 1 \
    --save-step 1 \
    --teacher-origin /type/of/teacher \             # Type of teacher: M2DS2/Finetuned
    --num-epochs 30 \
    --exp-save-path /path/to/save \
    --disable-tqdm
```
### MSDA (Proposed Method)
This approach implements the MSDA strategy as described in our paper to adapt the student model:
```bash
python msda.py \
    --src-train-dir /path/to/source/dataset \
    --trg-train-dir /path/to/target/dataset \
    --src-name source_name \                    # For logging and results
    --trg-name target_name \                    # For logging and results
    --batch-size 4 \
    --student-path facebook/wav2vec2-large-xlsr-53 \
    --student-spec-augment \                    # Optional: Apply SpecAugment on student
    --teacher-path /path/to/pretrained/teacher \
    --teacher-origin [m2ds2|finetuned] \        # Teacher type
    --log-step 1 \
    --save-step 1 \
    --num-epochs 30 \
    --delta 2e-5 \
    --gamma 1e-5 \
    --keep-percent 0.X \                        # Fraction of target data used
    --exp-save-path /path/to/save \
    --disable-tqdm
```
💡 Both meta_pl.py and msda.py save student and teacher models during training for future use or evaluation.
## Evaluation
To evaluate any trained model (fine-tuned, M2DS2, Meta-PL, MSDA):
```bash
python decode.py \
    --model-checkpoint /path/to/model \
    --model-name model_name \
    --dataset /path/to/dataset \
    --dataset-name dataset_name \
    --predictions-folder ./predictions \
    --batch-size 4 \
    --is-metapl   # Required for Meta-PL or MSDA checkpoints
```
## Citation
If you use this codebase, please consider citing the corresponding papers:

For M2DS2
```bibtex
```
For MSDA
```bibtex
```


## Contact
- d.damianos@athena.rc