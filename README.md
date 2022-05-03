# Emotion Recognition on SEED

This is the second assignment of the course CS7327 at ShangHai Jiao Tong University, 
which focuses on transfer learning and domain adaptation approaches in emotion recognition tasks. 
We use 5 out of 15 subjects in SJTU Emotion EEG Dataset(SEED), for more information, 
please visit [SEED Webpage](https://bcmi.sjtu.edu.cn/home/seed/index.html).


In this assignment, our work are summarized as follows:
- We setup baseline with traditional machine learning approaches in `tasks/baseline`;
- We adopt domain adaptation neural networks with `PyTorch` implementation in `tasks/DANN` ;
- We apply vanilla transfer learning paradigm to this task with `PyTorch` implementation in `tasks/vanilla_TL`;

## Setup environment

- First please set up environment in `requirements.txt`

## Baseline

- To run a baseline model, run the script `tasks/baseline/run.sh`
- Hyper-parameters can be configured in `tasks/baseline/conf/config.yaml`

## DANN

- To run a DANN model, run the script `tasks/DANN/run.sh`
- Hyper-parameters can be configured in `tasks/DANN/conf/config.yaml`

## Vanilla TL

- To run pre-training, run the script `tasks/vanilla_TL/run_backbone.sh`
- To train a classifier, run the script `tasks/vanilla_TL/run_classifier.sh`
- Hyper-parameters can be configured in `tasks/vanilla_TL/conf/config.yaml`
> Please note that the pre-trained checkpoints should by manually added to `task/vanilla_TL/backbone_checkpoints`, 
> You should copy the checkpoints from `hydra` outputs.

