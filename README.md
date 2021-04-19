# Speaker Verification Systems - Pytorch Implementation 

This project was stared from the [qqueing/DeepSpeaker-pytorch](https://github.com/qqueing/DeepSpeaker-pytorch). 

### 1. Datasets
Prepare data in kaldi way, and make features in Process_Data. 

- Development:
> Voxceleb1、Voxceleb2、Aishell1&2、CN-celeb1&2

- Augmentation:
> MUSAN、RIRS

- Test:
> SITW、Librispeech、TIMIT

####1.1 Pre-Processing

- Resample

- Butter Bandpass Filtering

- Augmentation

- [] LMS Filtering ( Defected )


### 2. Deep Speaker Verification Systems

#### 2.1 Neural Networks

- TDNN

The TDNN_v2 is implemented from 'https://github.com/cvqluu/TDNN/blob/master/tdnn.py'. The TDNN_v4 layer is implemented using nn.Conv2d. The TDNN_v5 layer implemented using nn.Conv1d
> ETDNN
>
> FTDNN
>
> DTDNN
>
> ARETDNN

- ResCNN

- LSTM 

LSTM and Attention-based LSTM

Input 40-dimensional MFCC.

- ResNet

ResNet34 with Fbank64.

#### 2.2 Loss Type

- A-Softmax

$T_x$

- AM-Softmax

- AAM-Softmax

- Center Loss

- Ring Loss

#### 2.3 Pooling Type

- Self-Attention

- Statistic Pooling

- Attention Statistic Pooling

### 3. Score

- Cosine

- PLDA


### 4. Neural Network Analysis

- Gradient

- Grad-CAM

- Grad-CAM++


#### . To do list
Work accomplished so far:

- [x] Models implementation
- [x] Data pipeline implementation - "Voxceleb"
- [x] Project structure cleanup.
- [ ] Trained simple ResNet10 with softmax+triplet loss for pre-training 10 batch and triplet loss for 18 epoch , resulted in accuracy ???
- [x] DET curve

#### Timeline
- [x] Extract x-vectors from trained Neural Network in 20190626
- [x] Code cleanup (factory model creation) 20200725
- [x] Modified preprocessing
- [x] Modified model for ResNet34,50,101 in 20190625
- [x] Added cosine distance in Triplet Loss(The previous distance is l2) in 20190703
- [ ] Adding scoring for identification
- [ ] Fork plda method for classification in python from: https://github.com/RaviSoji/plda/blob/master/plda/

### 5. Performance
#### 5.1 Baseline
|Stage|Model|epoch|Loss Type|Loss value|Accuracy (%) Train/Test|EER (%)| 
|:--------:|:------------:|:---:|:--------------:|:--------------:|:----------------:|:-------:|
|1| Resnet-10    |1:22 |Triplet | 6.6420 ~ 0.0113 | 0.8553/0.8431  |... |
| | ResNet-34    |1:8  |Triplet | 8.0285:0.0301 | 0.8360/0.8302  | ... |
|2| TDNN         |....| Softmax | 8.0285:0.0301 | 0.8360/0.8302  | ... |
| | ETDNN         |....| Softmax | 8.0285:0.0301 | 0.8360/0.8302  | ... |
| | FTDNN         |....| Softmax | 8.0285:0.0301 | 0.8360/0.8302  | ... |
| | DTDNN         |....| Softmax | 8.0285:0.0301 | 0.8360/0.8302  | ... |
| | ARETDNN         |....| Softmax | 8.0285:0.0301 | 0.8360/0.8302  | ... |
|3| LSTM         |....|     ...   |        ...        |   ...  | ... |
| | LSTM         |....| Softmax | 8.0285:0.0301 | 0.8360/0.8302  | ... |
| | LSTM         |....| Softmax | 8.0285:0.0301 | 0.8360/0.8302  | ... |
|2| ...          |....|    ...    | ...   |   ...   | ... |
|2| TDNN         |....| Softmax | 8.0285:0.0301 | 0.8360/0.8302  | ... |

#### 5.2 Baseline

### 6. Reference:  

> [1] Cai, Weicheng, Jinkun Chen, and Ming Li. "Analysis of Length Normalization in End-to-End Speaker Verification System.." conference of the international speech communication association (2018): 3618-3622.
>
> [2] ...














