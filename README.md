# Reimplementation of Severyn and Moschitti's CNN for Ranking Short Text Pairs

**NOTE: This repo contains code for the original Torch implementation from SIGIR 2017. The code is not being maintained anymore and has been superseded by a PyTorch reimplementation in [Castor](https://github.com/castorini/Castor). This repo exists solely for archival purposes.**

This repo contains code for the reproduction of Severyn and Moschitti's CNN for Ranking Short Text Pairs from SIGIR 2015 [1]. Our findings were published in a SIGIR 2017 short paper [2].

Getting Started
-----------
``1.`` Please install the Torch library by following instructions here: https://github.com/torch/distro

``2.`` Using following script to download and preprocess the Glove word embedding:
```
$ sh fetch_and_preprocess.sh
``` 
Please make sure your python version >= 2.7, otherwise you will encounter an exception when unzip the downloaded embedding file.

``3.`` Before you run our model, please set the number of threads >= 5 for parallel processing. 
```
$ export OMP_NUM_THREADS=5
```

Running
--------
Run the original SIGIR'15 model with external IDF features on TrecQA raw dataset:

$ ``th trainSIGIR15.lua -dataset TrecQA -version raw -train train-all -model conv -sim bilinear -ext``

Run linear model with external features:

$ ``th trainSIGIR15.lua -dataset TrecQA -model linear -ext``

All model options described in our paper [2] can be specified through the above command line parameters.

Reference 
---------

[1] Aliaksei Severyn and Alessandro Moschitti. [Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks](https://dl.acm.org/citation.cfm?id=2767738). SIGIR 2015.

[2] Jinfeng Rao, Hua He, and Jimmy Lin. [Experiments with Convolutional Neural Network Models for Answer Selection](https://dl.acm.org/citation.cfm?id=3080648). SIGIR 2017.
