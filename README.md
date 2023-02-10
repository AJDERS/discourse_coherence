# Danish Discourse Coherence Classification
This repository is for training a classifier for danish discourse coherence. It uses the (DDisco)[https://huggingface.co/datasets/ajders/ddisco] dataset.

## Dataset Description

The DDisco dataset is a dataset which can be used to train models to classify levels of coherence in _danish_ discourse. Each entry in the dataset is annotated with a discourse coherence label (rating from 1 to 3):

1: low coherence (difficult to understand, unorganized, contained unnecessary details and can not be summarized briefly and easily)
2: medium coherence
3: high coherence (easy to understand, well organized, only contain details that support the main point and can be summarized briefly and easily).
Grammatical and typing errors are ignored (i.e. they do not affect the coherency score) and the coherence of a text is considered within its own domain.

## Model Description

This model is a fine-tuned version of [NbAiLab/nb-bert-base](https://huggingface.co/NbAiLab/nb-bert-base) on the [DDisco](https://huggingface.co/datasets/ajders/ddisco) dataset.
It achieves the following results on the evaluation set:
- Loss: 0.7487
- Accuracy: 0.6915

### Training procedure

#### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 4
- eval_batch_size: 4
- seed: 703
- gradient_accumulation_steps: 16
- total_train_batch_size: 64
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.05
- num_epochs: 6.0

#### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 1.3422        | 0.4   | 5    | 1.0166          | 0.5721   |
| 0.9645        | 0.8   | 10   | 0.8966          | 0.5721   |
| 0.9854        | 1.24  | 15   | 0.8499          | 0.5721   |
| 0.8628        | 1.64  | 20   | 0.8379          | 0.6517   |
| 0.9046        | 2.08  | 25   | 0.8228          | 0.5721   |
| 0.8361        | 2.48  | 30   | 0.7980          | 0.5821   |
| 0.8158        | 2.88  | 35   | 0.8095          | 0.5821   |
| 0.8689        | 3.32  | 40   | 0.7989          | 0.6169   |
| 0.8125        | 3.72  | 45   | 0.7730          | 0.6965   |
| 0.843         | 4.16  | 50   | 0.7566          | 0.6418   |
| 0.7421        | 4.56  | 55   | 0.7840          | 0.6517   |
| 0.7949        | 4.96  | 60   | 0.7531          | 0.6915   |
| 0.828         | 5.4   | 65   | 0.7464          | 0.6816   |
| 0.7438        | 5.8   | 70   | 0.7487          | 0.6915   |


#### Framework versions

- Transformers 4.26.0
- Pytorch 1.13.0a0+d0d6b1f
- Datasets 2.9.0
- Tokenizers 0.13.2

### Additional Information

[DDisCo: A Discourse Coherence Dataset for Danish](https://aclanthology.org/2022.lrec-1.260.pdf)

### Contributions

[@ajders](https://github.com/ajders)