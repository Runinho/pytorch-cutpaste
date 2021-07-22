# Implementation of CutPaste

This is a **unofficial** work in progress PyTorch reimplementation of [CutPaste: Self-Supervised Learning for Anomaly Detection and  Localization](https://arxiv.org/abs/2104.04015) and in no way affiliated with the original authors. Use at own risk. Pull requestes and feedback is appreciated.

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
conda create -n cutpaste pytorch torchvision torchaudio cudatoolkit=10.2 seaborn pandas tqdm tensorboard scikit-learn -c pytorch
conda activate cutpaste
```

## Run Training
```
python run_training.py --model_dir models --head_layer 2
```
The Script will train a model for each defect type and save it in the `model_dir` Folder.

One can track the training progress of the models with tensorboard:
```
tensorboard --logdir logdirs
```

## Run Evaluation
```
python eval.py --model_dir models --head_layer 2
```
This will create a new directory `Eval` with plots for each defect type/model.

## Some implementation details
Only the normal CutPaste augmentation and 2-Class classification variant is implemented.

The pasted image patch always origins from the same image it is pasted to. I'm not sure if this is a Problem and if this is also the case in the original paper/code.

# TODOs
- [x] implement Cut-Paste Scar
- [ ] implement gradCam
- [ ] implement localization variant
- [ ] add option to finetune on EfficientNet(B4)
- [ ] clean up parameters and move them into the arguments of the scripts
- [ ] compare results of this reimplementation with the results of the paper
