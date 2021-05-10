# Implementation of CutPaste

This is a work in progress PyTorch implementation of [CutPaste: Self-Supervised Learning for Anomaly Detection and  Localization](https://arxiv.org/abs/2104.04015).

## Setup
Download the MVTec Anomaly detection Dataset from [here](https://www.mvtec.com/company/research/datasets/mvtec-ad) and extract it into a new folder named `Data`.

Install the following requirements:
1. Pytorch and torchvision
2. sklearn
3. pandas
4. seaborn
5. tqdm
6. tensorboard

For example with [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html):
```
conda crate -n cutpaste pytorch torchvision torchaudio cudatoolkit=10.2 seaborn pandas tqdm tensorboard -c pytorch
conda activate cutpaste
```

## Run Training
```
python run_training.py --model_dir models --head_layer 2
```
The Script will train a model for each defect type and save it in the `model_dir` Folder.

## Run Evaluation
```
python eval.py --model_dir models --head_layer 2
```
This will create a new directory `Eval` with plots for each defect type/model.
