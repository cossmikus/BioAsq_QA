# Bio-medical Question Answering

## Abstract
Fine-tuning a pre-trained T5 model on a biomedical question-answering dataset, such as BioASQ, for a Question Answering task (evaluation based on BLEU), followed by deployment of the model to Hugging Face.

## Introduction
The project's task is to fine-tune a pre-trained T5 model on a biomedical question-answering dataset, specifically BioASQ, for a Question Answering task. The evaluation of the model will be based on BLEU score. Following successful fine-tuning, the model will be deployed to Hugging Face.

## Dataset and Methodology
The project utilizes a dataset consisting of 3266 rows of training data from the BioASQ dataset. The dataset is divided into "question", "answer" and "context" columns. A T5 tokenizer is employed to tokenize the data for model training. Torch tensor tokenizer returns "input_ids", "attention_mask" and "labels" subcomponents that further will be used for training.

Fine-tuning of the T5 model is conducted over 2 epochs using the Adam optimizer. After each epoch, the model is trained using `model.train()` and evaluated using `model.eval()`. Learning rate in optimizer is set to 10^-5. For the batches, the optimizer is used, and for loss backpropagation is employed.

## Results
The model's performance is evaluated using BLEU score, achieving a score of 69% on a subset of 500 rows from the dataset.

## Deployment
Upon successful fine-tuning and evaluation, the trained model and tokenizer are deployed to Hugging Face under the repository "starman76/t5_500". Deployment is achieved using the following commands:

