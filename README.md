# Speaker Recognition Systems - Pytorch Implementation

This project was forked and started from
the [qqueing/DeepSpeaker-pytorch](https://github.com/qqueing/DeepSpeaker-pytorch).

### 1. Datasets
Prepare data in kaldi way, and make features in Process_Data. 

- Development:

> Voxceleb1、Voxceleb2、Aishell1&2、CN-celeb1&2、aidatatang_200zh、MAGICDATA、TAL_CSASR
> ChiME5、VOiCES、CommonVoice、AMI

- Augmentation:
> MUSAN、RIRS

- Test:
> SITW、Librispeech、TIMIT

####1.1 Pre-Processing

- Resample

- Butter Bandpass Filtering

- Augmentation

- [ ] LMS Filtering ( Defected )


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
> Aggregated-Residual TDNN
>
> ECAPA TDNN

- ResCNN

- LSTM

LSTM and Attention-based LSTM

Input 40-dimensional MFCC.

- ResNet

ResNet34

#### 2.2 Loss Type

##### Classification

- A-Softmax

- AM-Softmax

- AAM-Softmax

- Center Loss

- Ring Loss

##### End-toEnd

- Generalized End-to-End Loss

- Triplet Loss

- Contrastive Loss

- Prototypical Loss

- Angular Prototypical Loss

#### 2.3 Pooling Type

- Self-Attention

- Statistic Pooling

- Attention Statistic Pooling

- GhostVALD Pooling

### 3. Score

- Cosine

- PLDA

- DET

- t-sne


### 4. Neural Network Analysis

- Gradient

- Grad-CAM

- Grad-CAM++

### . To do list
Work accomplished so far:

- [x] Models implementation
- [x] Data pipeline implementation - "Voxceleb"
- [x] Project structure cleanup.
- [ ] Trained simple ResNet10 with softmax+triplet loss for pre-training 10 batch and triplet loss for 18 epoch , resulted in accuracy ???
- [x] DET curve

### Timeline

- [x] Extract x-vectors from trained Neural Network in 20190626
- [x] Code cleanup (factory model creation) 20200725
- [x] Modified preprocessing
- [x] Modified model for ResNet34,50,101 in 20190625
- [x] Added cosine distance in Triplet Loss(The previous distance is l2) in 20190703
- [ ] Adding scoring for identification
- [ ] Fork plda method for classification in python from: https://github.com/RaviSoji/plda/blob/master/plda/

### 5. Performance

#### 5.1 Baseline

|   Group  |      Model   |  epoch  |    Loss Type   |       Loss      |    Train/Test   |    Accuracy (%)  |  EER (%) |
|:--------:|:------------:|:-------:|:--------------:|:---------------:|:----------------:|:----------------:|:--------:|
|1         | Resnet-10    |  1:22   |    Triplet     |  6.6420:0.0113  |  0.8553/0.8431  |    ...   |
|          | ResNet-34    |   1:8   |  CrossEntropy  |  8.0285:0.0301  |  0.8360/0.8302  |    ...   |
|2         | TDNN         |    40   |  CrossEntropy  |  3.1716:0.0412  |  vox1 dev/test  | 99.9994/99.5871  | 1.6700/5.4030 |
|2         | TDNN         |    40   |  CrossEntropy  |  3.0382:0.2196  | vox2 dev/vox1 test | 98.5265/98.2733 | 3.0800/3.0859 |
| | ETDNN         |....| Softmax | 8.0285:0.0301 | 0.8360/0.8302  | ... | |
| | FTDNN         |....| Softmax | 8.0285:0.0301 | 0.8360/0.8302  | ... | |
| | DTDNN         |....| Softmax | 8.0285:0.0301 | 0.8360/0.8302  | ... | |
| | ARETDNN         |....| Softmax | 8.0285:0.0301 | 0.8360/0.8302  | ... | |
|3| LSTM         |....|     ...   |        ...        |   ...  | ... | |
| | LSTM         |....| Softmax | 8.0285:0.0301 | 0.8360/0.8302  | ... | |
| | LSTM         |....| Softmax | 8.0285:0.0301 | 0.8360/0.8302  | ... | |
|2| ...          |....|    ...    | ...   |   ...   | ... |
|2| TDNN         |....| Softmax | 8.0285:0.0301 | 0.8360/0.8302  | ... | |

- TDNN_v5, Training set: voxceleb 2 161-dimensional spectrogram, Loss: arcosft, Cosine Similarity

  |   Test Set      |    EER ( % )  |   Threshold   |  MinDCF-0.01   |   MinDCF-0.01  |     Date     |
    |:---------------:|:-------------:|:-------------:|:--------------:|:--------------:|:------------:|
  |   vox1 test     |    2.3542%    |   0.2698025   |    0.2192      |     0.2854     |   20210426   |
  |   sitw dev      |    2.8109%    |   0.2630014   |    0.2466      |     0.4026     |   20210515   |
  |   sitw eval     |    3.2531%    |   0.2642460   |    0.2984      |     0.4581     |   20210515   |
  |  cnceleb test   |   16.8276%    |   0.2165570   |    0.6923      |     0.8009     |   20210515   |
  |  aishell2 test  |   10.8300%    |   0.2786811   |    0.8212      |     0.9527     |   20210515   |
  |   aidata test   |   10.0972%    |   0.2952531   |    0.7859      |     0.9520     |   20210515   |

#### 5.2 Baseline

### 6. Reference:

> [1] Cai, Weicheng, Jinkun Chen, and Ming Li. "Analysis of Length Normalization in End-to-End Speaker Verification System.." conference of the international speech communication association (2018): 3618-3622.
>
> [2] ...














