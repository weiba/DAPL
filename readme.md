DAPL
===============================
Source code and data for "Predicting clinical anticancer drug response of patient by using domain alignment and prototypical learning"

![Framework of DAPL](https://github.com/weiba/DAPL/workflow.jpg)  
# Requirements
- python==3.10.8
- pytorch==1.13.1
- torch_geometric==2.3.1
- numpy==1.21.5+mkl
- scipy==1.7.3
- pandas==1.3.5
- scikit-learn=1.0.2
- hickle==5.0.2
- rdkit==2023.3.1
- networkx==2.6.3
# Usage
- python train.py {PDTC and TCGA predictio task}


The *train.py* file accomplishes two prediction tasks, including prediction of the PDTC and TCGA datasets, you can also modify the main functions to predict only one dataset. The train folder contains the model file model.py and some data processing functions dataprocess.py.  All functions are developed using PyTorch and support CUDA.

# Contact
If you have any question regard our code or data, please do not hesitate to open a issue or directly contact me (weipeng1980@gmail.com).
