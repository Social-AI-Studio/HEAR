# Baseline Experiments
## Data Preparation
### Training Set
* The whole dataset is too large for training
    * Not neccessary - too many small propagations
* Sampling
    * Sampling based on label
    * Preserve data of label 2, 3, 4 (high hatefulness, small amount of propagations)
    * Downsamples data of label 0, 1
### Test Set
* Keep stratified test set
### Features
* ~~For each tweet, use TFIDF vector of top 2048 global frequent words in the datasets as its feature~~
* Use fasttext embedding trained on COVID-19 related data
    * A RNN model is pre-trained on hateful speech classification task to generate tweet-level feature


## Model
### Main Model
* Based on [Rumor Detection on Twitter with Tree-structured Recursive Neural Networks](https://www.aclweb.org/anthology/P18-1184/) (ACL'18)
* Top-down and bottom-up tree-structured recursive neural network
    * GRU as node operator
* For bottom-up model, a varaint with attention layer is added
### Classifier
* After getting the embedding of a propagation
* ~~2 layer MLP, output dimension is 4~~
    * ~~Better encode the relation between labels~~
    * ~~Output [0, 0, 0, 0] classified as label 0~~
    * ~~Output [1, 0, 0, 0] classified as label 1, etc~~

## Experiment Setting
### Models
* Top-down backbone + classifier (TD Net)
* Bottum-up backbone + classifier (BU Net)
* Bottum-up attention backbone + classifier (BU attn Net)
* Concat top-down and bottom-up features + classifier (DD Net)
* Concat top-down and bottom-up attention features + classifier (DD attn Net)

### Parameters
* SGD optimizer
    * lr = 0.002
    * momentum = 0.9
    * weight_decay = 1e-4
* CE loss
    * ~~weight = [1., 2., 3., 3.]~~
* Hidden size = 256
* Epochs = 10

## Result
### GeorgiaTech dataset
#### TD Net
* F1 score
    * Train: 0.9065
    * Test:  0.4495
#### BU Net
* F1 score
    * Train: 0.7519
    * Test:  0.4715
#### DD Net
* F1 score
    * Train: 0.8448
    * Test: 0.4148
### COVID-Hate dataset
* Report balanced accuracy (i.e., average of class recall) and f1 score (averaged across classes)
#### BU Net
* f1: 0.614800764862369
* acc: 0.6134094488082552
#### BU Net with attn
* f1: 0.6186547740499815
* acc: 0.622167371005253
#### TD Net
* f1: 0.5578762863195781
* acc: 0.5724126594541262
#### DD Net
#### DD Net with attn
* f1: 0.6035906984035163
* acc: 0.622819134749524
