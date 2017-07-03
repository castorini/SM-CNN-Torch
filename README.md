# Reproduction
This is the repo for the reproduction Torch experiments of the SIGIR'15 convolutional model [1]. More detailed and interesting findings can be found in our paper [2]. 

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
``[1]. Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks, Aliaksei Severyn and Alessandro Moschi, SIGIR 2015`` 

``[2]. Experiments with Convolutional Neural Network Models for Answer Selection, Jinfeng Rao, Hua He, and Jimmy Lin, SIGIR 2017``
