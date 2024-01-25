DAPL
===============================
Source code and data for "Predicting clinical anticancer drug response of patient by using domain alignment and prototypical learning"

![Framework of DAPL](https://github.com/weiba/DAPL/blob/master/workflow.jpg)  
# Requirements
All implementations of DAPL are based on PyTorch. DAPL requires the following dependencies:
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
# Data
- Data defines the data used by the model
	- data/TCGA records training data, test data, and labeling related to the five drugs associated with TCGA.
	- data/PDTC records training data, test data, and labeling related to the fifty drugs associated with PDTC.
	- data/ccle_sample_info.csv records biological information related to CCLE samples.
	- data/pretrain_ccle.csv records gene expression data from unlabeled CCLE samples.
	- data/pretrain_tcga.csv records gene expression data from unlabeled TCGA samples.
	- data/pdtc_uq1000_feature.csv records gene expression data from unlabeled PDTC samples.
	- data/GDSC1_fitted_dose_response_25Feb20.csv and data/GDSC2_fitted_dose_response_25Feb20.csv records data on drug use and response in GDSC samples.
	- data/DrugResponsesAUCModels.txt records response data for PDTC sample-drug pairs. 
	- data/pdtc_gdsc_drug_mapping.csv records the 50 drug names associated with pdtc and their smiles.
	- data/smile_inchi326.csv records data on 326 drug molecular fingerprints used for pre-training.
	- data/uq1000_feature.csv records gene expression data for unlabeled TCGA samples and CCLE samples.
	- data/xena_sample_info_df.csv records biological information related to TCGA samples.
- data.py defines the data loading of the model.
- train.py defines the training of the model.
- precontext.py defines the pre-training of the drug GNN.
- tools/dataprocess.py defines some code for data processing.
- tools/model.py defines the model used in the training process.
- drugmodels folder defines some drug GNN models.In our code we use ginconv.

## Preprocessing your own data
Explanations on how you can process your own data and prepare it for DAPL running.
> In our study, the source cell line data and targe partient data we followed are from [codeae](https://codeocean.com/capsule/1993810/tree/v1)[1]. You can run our program with your own data, and process the data you use into source and target domain data of the same dimensions, while just having a one-dimensional labeled data for each sample-drug pair data. You can refer to the style of the data in data/TCGA.
> 
> [1] He, Di, et al. "A context-aware deconfounding autoencoder for robust prediction of personalized clinical drug response from cell-line compound screening." Nature Machine Intelligence 4.10 (2022): 879-892.

# Usage
Once you have configured the environment, you can simply run DAPL by running command:
```
python train.py
```
> The entire training of the model, pre-training of the drug model just needs to run it. The training of the PDTC dataset as well as the TCGA dataset is included in the main function in ***train.py***. Our training process includes a pre-training process using unlabeled data, as well as a training process for individual cancer drug related sample data. During the training process we divided the source domain data into five folds and used the target domain as the test data. 
The *train.py* file accomplishes two prediction tasks, including prediction of the PDTC and TCGA datasets, you can also modify the main function to predict only one dataset. The train folder contains the model file model.py and some data processing functions dataprocess.py.  All functions are developed using PyTorch and support CUDA.

> The evaluation metrics include AUC, AUPRC. If you want to evaluate the test set results, you can use the following command:
```
python evaluate.py
```

# Contact
If you have any question regard our code or data, please do not hesitate to open a issue or directly contact me (weipeng1980@gmail.com).
