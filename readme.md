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
- pretrain.py defines the pre-training of the model.
- precontext.py defines the pre-training of the drug GNN.
- prototypical.py defines the classifier training step of the model.
- tools/dataprocess.py defines some code for data processing.
- tools/model.py defines the model used in the training process.
- drugmodels folder defines some drug GNN models.In our code we use ginconv.

## Preprocessing your own data
Explanations on how you can process your own data and prepare it for DAPL running.
> In our study, the source cell line data and targe partient data we followed are from [codeae](https://codeocean.com/capsule/1993810/tree/v1)[1]. You can run our program with your own data, and process the data you use into source and target domain data of the same dimensions, while just having a one-dimensional labeled data for each sample-drug pair data. You can refer to the style of the data in data/TCGA.
> 
> [1] He, Di, et al. "A context-aware deconfounding autoencoder for robust prediction of personalized clinical drug response from cell-line compound screening." Nature Machine Intelligence 4.10 (2022): 879-892.

# Usage
Once you have configured the environment, you can simply run **DAPL** in 3 steps using the data we provided:
```
1. python pretrain.py
2. python precontext.py
3. python prototypical.py
```
Or you can just run the following line of code in place of the three above to replicate our experimental results using our data:
```
python train.py
```
To bulid and evaluate our model, we uses cell line gene expression data as the source domain and patient gene expression data as the target domain.
Meanwhile, we use 5-fold cross-validation, we use StratifiedKFold to divide the source domain data into training set: validation set = 4:1, and we finally use the AUC and AUPRC averaged over fifty percent discounts of the test set as the evaluation criteria for the model.

>In **pretrain.py** we perform a parameter search and save the pre-trained model. We use a cosine
annealing strategy with the learning rate set to 0.001 and a grid
search to store the model under different pretraining epochs,
while we use an early stopping strategy for the model when the
loss of the validation set does not decrease for 20 consecutive
times
>
>In **precontext.py** we build the molecular map using drug smiles and train the drug feature extractor.We set the learning rate
to 0.01 and the training epochs to 100.
>
>In **prototypical.py** we perform prototypical learning and extract features using the pre-trained encoder and drug GNN while we perform parameter search. 
> We train the classifier at this stage by performing a parameter search, testing for the use of a cosine annealing strategy while lr=[0.01, 0.001].
> We predict model performance at this stage.

Alternatively, you can run our program with your own data and some other settings as follows:
```
1. python pretrain.py \
--outfolder path/to/folder_to_save_pretrain_models \
--source path/to/your_pretrain_source_data.csv \
--target path/to/your_pretrain_target_data.csv

2. python precontext.py
--dataset path/to/your_drug_smiles.csv \
--out path/to/save_drugpth.pth

3. python prototypical.py \
--dataset other \
--data path/to/your_data_folder \
--drug path/to/your_drug_smiles.csv \
--pretrain_model path/to/your_pretrain_models_path \
--outfolder path/to/save_result_and_others \
--outname result_file_name.csv 
```
Note: 
>You need to ensure that the data dimensions of your 
source and target domains are the same.
> 
>The **your_drug_smiles.csv** requires the first column to be the name of the drug, with a column named 'smiles'
>
>The **your_data_folder** is a folder that contains many medication folders while each medication folder contains sourcedata.csv,targetdata.csv,sourcelabel.csv,targetlabel.csv.The format of each file can be referred to. /data/TCGA.


In addition we provide a separate test code to evaluate our own dataset:
```
python evaluate.py
```

# Contact
If you have any question regard our code or data, please do not hesitate to open a issue or directly contact me (weipeng1980@gmail.com).
