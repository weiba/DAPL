import pandas as pd
import numpy as np
import os
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('use device:', device)
from torch.utils.data import TensorDataset, DataLoader

def pretrain_data():
    ccle_df = pd.read_csv(os.path.join('data', 'pretrain_ccle.csv'), index_col=0, header=0)
    xena_df = pd.read_csv(os.path.join('data', 'pretrain_tcga.csv'), index_col=0, header=0)
    ccle_sample_info_df = pd.read_csv(os.path.join('data', 'ccle_sample_info_df.csv'), index_col=0, header=0)
    xena_sample_info_df = pd.read_csv(os.path.join('data', 'xena_sample_info_df.csv'), index_col=0, header=0)

    excluded_ccle_samples = []
    excluded_ccle_samples.extend(ccle_df.index.difference(ccle_sample_info_df.index))
    excluded_ccle_diseases = ccle_sample_info_df.primary_disease.value_counts()[
        ccle_sample_info_df.primary_disease.value_counts() < 2].index
    excluded_ccle_samples.extend(
        ccle_sample_info_df[ccle_sample_info_df.primary_disease.isin(excluded_ccle_diseases)].index)
    to_split_ccle_df = ccle_df[~ccle_df.index.isin(excluded_ccle_samples)]
    train_ccle_df, test_ccle_df = train_test_split(to_split_ccle_df, test_size=0.1,
                                                stratify=ccle_sample_info_df.loc[
                                                    to_split_ccle_df.index].primary_disease)
    test_ccle_df = test_ccle_df.append(ccle_df.loc[excluded_ccle_samples])
    train_xena_df, test_xena_df = train_test_split(xena_df, test_size=len(test_ccle_df) / len(xena_df),
                                                   stratify=xena_sample_info_df['_primary_disease'],
                                                   random_state=2020)
    # # gene function data:DATA.Data num:1284
    # gene_function_adj_df = pd.read_csv('gene_function_adj.csv', index_col=0, header=0)
    # # gene function adj 1284->1426
    # gene_list = ccle_df.columns.to_list()
    # gene_function_adj_df = gene_function_adj_df.reindex(index=gene_list, columns=gene_list, fill_value=0)
    # gene_function_adj_df = gene_function_adj_df.loc[gene_list, gene_list]
    # gene_function_adj_tensor = torch.from_numpy(gene_function_adj_df.values).type(torch.float32).to(device)
    # row_indices, col_indices = torch.where(gene_function_adj_tensor != 0)
    # edge_index = torch.stack([row_indices, col_indices], dim=0).to(gene_function_adj_tensor.device)
    # create dataloader
    ccle_tensor = torch.from_numpy(ccle_df.values).type(torch.float32).to(device)
    ccle_test_tensor = torch.from_numpy(test_ccle_df.values).type(torch.float32).to(device)
    tcga_tensor = torch.from_numpy(xena_df.values).type(torch.float32).to(device)
    tcga_test_tensor = torch.from_numpy(test_xena_df.values).type(torch.float32).to(device)

    # dataloader
    batch_size = 64
    ccleDataset = TensorDataset(ccle_tensor)
    ccleloader = DataLoader(ccleDataset, batch_size=batch_size, shuffle=True, drop_last=True)

    tcgaDataset = TensorDataset(tcga_tensor)
    tcgaloader = DataLoader(tcgaDataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return (ccleloader, ccle_test_tensor), (tcgaloader, tcga_test_tensor)


def PDTC_source_5fold(drug):
    measurement = 'Z_SCORE'
    threshold = 0.0
    drugs_to_keep = [drug.lower()]
    gdsc_target_file1 = os.path.join('data', 'GDSC1_fitted_dose_response_25Feb20.csv')
    gdsc_target_file2 = os.path.join('data', 'GDSC2_fitted_dose_response_25Feb20.csv')
    gdsc1_response = pd.read_csv(gdsc_target_file1)
    gdsc2_response = pd.read_csv(gdsc_target_file2)
    gdsc1_sensitivity_df = gdsc1_response[['COSMIC_ID', 'DRUG_NAME', measurement]]
    gdsc2_sensitivity_df = gdsc2_response[['COSMIC_ID', 'DRUG_NAME', measurement]]
    gdsc1_sensitivity_df.loc[:, 'DRUG_NAME'] = gdsc1_sensitivity_df['DRUG_NAME'].str.lower()
    gdsc2_sensitivity_df.loc[:, 'DRUG_NAME'] = gdsc2_sensitivity_df['DRUG_NAME'].str.lower()

    if measurement == 'LN_IC50':
        gdsc1_sensitivity_df.loc[:, measurement] = np.exp(gdsc1_sensitivity_df[measurement])
        gdsc2_sensitivity_df.loc[:, measurement] = np.exp(gdsc2_sensitivity_df[measurement])

    gdsc1_sensitivity_df = gdsc1_sensitivity_df.loc[gdsc1_sensitivity_df.DRUG_NAME.isin(drugs_to_keep)]
    gdsc2_sensitivity_df = gdsc2_sensitivity_df.loc[gdsc2_sensitivity_df.DRUG_NAME.isin(drugs_to_keep)]
    gdsc1_target_df = gdsc1_sensitivity_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()
    gdsc2_target_df = gdsc2_sensitivity_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()
    gdsc1_target_df = gdsc1_target_df.loc[gdsc1_target_df.index.difference(gdsc2_target_df.index)]
    gdsc_target_df = pd.concat([gdsc1_target_df, gdsc2_target_df])
    target_df = gdsc_target_df.reset_index().pivot_table(values=measurement, index='COSMIC_ID', columns='DRUG_NAME')
    ccle_sample_file = os.path.join('11.9mypdtctest', 'ccle_sample_info.csv')
    ccle_sample_info = pd.read_csv(ccle_sample_file, index_col=4)
    ccle_sample_info = ccle_sample_info.loc[ccle_sample_info.index.dropna()]
    ccle_sample_info.index = ccle_sample_info.index.astype('int')
    gdsc_sample_file = os.path.join('11.9mypdtctest', 'gdsc_cell_line_annotation.csv')
    gdsc_sample_info = pd.read_csv(gdsc_sample_file, header=0, index_col=1)
    gdsc_sample_info = gdsc_sample_info.loc[gdsc_sample_info.index.dropna()]
    gdsc_sample_info.index = gdsc_sample_info.index.astype('int')
    # gdsc_sample_info = gdsc_sample_info.loc[gdsc_sample_info.iloc[:, 8].dropna().index]
    gdsc_sample_mapping = gdsc_sample_info.merge(ccle_sample_info, left_index=True, right_index=True, how='inner')[
        ['DepMap_ID']]
    gdsc_sample_mapping_dict = gdsc_sample_mapping.to_dict()['DepMap_ID']
    target_df.index = target_df.index.map(gdsc_sample_mapping_dict)
    target_df = target_df.loc[target_df.index.dropna()]
    ccle_target_df = target_df[drugs_to_keep[0]]
    ccle_target_df.dropna(inplace=True)
    gex_feature_file = os.path.join('data', 'uq1000_feature.csv')
    gex_features_df = pd.read_csv(gex_feature_file, index_col=0)
    ccle_labeled_samples = gex_features_df.index.intersection(ccle_target_df.index)

    if threshold is None:
        threshold = np.median(ccle_target_df.loc[ccle_labeled_samples])

    ccle_labels = (ccle_target_df.loc[ccle_labeled_samples] < threshold).astype('int')
    ccle_labeled_feature_df = gex_features_df.loc[ccle_labeled_samples]
    assert all(ccle_labels.index == ccle_labeled_feature_df.index)
    s_kfold = StratifiedKFold(n_splits=5, random_state=2020, shuffle=True)
    for train_index, test_index in s_kfold.split(ccle_labeled_feature_df.values, ccle_labels.values):
        train_labeled_ccle_df, test_labeled_ccle_df = ccle_labeled_feature_df.values[train_index], \
                                                      ccle_labeled_feature_df.values[test_index]
        train_ccle_labels, test_ccle_labels = ccle_labels.values[train_index], ccle_labels.values[test_index]
        # df->tensor
        ccle_train_data = torch.from_numpy(train_labeled_ccle_df).type(torch.float32).to(device)
        ccle_train_label = torch.from_numpy(train_ccle_labels).type(torch.float32).squeeze().to(device)
        ccle_test_data = torch.from_numpy(test_labeled_ccle_df).type(torch.float32).to(device)
        ccle_test_label = torch.from_numpy(test_ccle_labels).type(torch.float32).squeeze().to(device)

        yield (ccle_train_data, ccle_train_label), (ccle_test_data, ccle_test_label)

def PDTC_target_data(drug):
    pdtc_gex_file = os.path.join('data', 'pdtc_uq1000_feature.csv')
    pdtc_features_df = pd.read_csv(pdtc_gex_file, index_col=0)
    pdtc_target_file = os.path.join('data', 'DrugResponsesAUCModels.txt')
    target_df = pd.read_csv(pdtc_target_file, index_col=0, sep='\t')
    drug_target_df = target_df.loc[target_df.Drug == drug]
    labeled_samples = drug_target_df.index.intersection(pdtc_features_df.index)
    drug_target_vec = drug_target_df.loc[labeled_samples, 'AUC']
    drug_feature_df = pdtc_features_df.loc[labeled_samples]
    threshold = np.median(drug_target_vec)
    drug_label_vec = (drug_target_vec < threshold).astype('int')
    tcga_features = torch.from_numpy(drug_feature_df.values).type(torch.float32).to(device)
    tcga_label = torch.from_numpy(drug_label_vec.values).type(torch.float32).squeeze().to(device)
    return (tcga_features, tcga_label)

def PDTC_data_generator(drug):
    drug_mapping_df = pd.read_csv(os.path.join('data', 'pdtc_gdsc_drug_mapping.csv'), index_col=0)
    drug_name = drug_mapping_df.loc[drug, 'drug_name']
    tcga_data = PDTC_target_data(drug_name)
    gdsc_drug = drug_mapping_df.loc[drug, 'gdsc_name']
    ccle_data_tuple = PDTC_source_5fold(gdsc_drug)
    for ccle_train_data, ccle_eval_data in ccle_data_tuple:
        yield (ccle_train_data, ccle_eval_data, tcga_data)


def TCGA_source_5fold(drug):
    # data df gene_num:1426
    ccle_features_df = pd.read_csv(os.path.join('data','TCGA', drug + 'data', 'ccledata.csv'), index_col=0, header=0)
    ccle_label_df = pd.read_csv(os.path.join('data','TCGA', drug + 'data', 'cclelabel.csv'), index_col=0, header=0)
    # split 5-fold
    s_kfold = StratifiedKFold(n_splits=5, random_state=2020, shuffle=True)
    for train_index, test_index in s_kfold.split(ccle_features_df.values, ccle_label_df.values):
        train_labeled_ccle_df, test_labeled_ccle_df = ccle_features_df.values[train_index], \
                                                    ccle_features_df.values[test_index]
        train_ccle_labels, test_ccle_labels = ccle_label_df.values[train_index], ccle_label_df.values[test_index]
        # df->tensor
        ccle_train_data = torch.from_numpy(train_labeled_ccle_df).type(torch.float32).to(device)
        ccle_train_label = torch.from_numpy(train_ccle_labels).type(torch.float32).squeeze().to(device)
        ccle_test_data = torch.from_numpy(test_labeled_ccle_df).type(torch.float32).to(device)
        ccle_test_label = torch.from_numpy(test_ccle_labels).type(torch.float32).squeeze().to(device)

        yield (ccle_train_data, ccle_train_label), (ccle_test_data, ccle_test_label)


def TCGA_target_data(drug):
    tcga_features_df = pd.read_csv(os.path.join('data','TCGA', drug + 'data', 'tcgadata.csv'), index_col=0, header=0)
    # tcga_features_df = tcga_features_df.reindex(columns=ccle_columns)
    tcga_label_df = pd.read_csv(os.path.join('data','TCGA', drug + 'data', 'tcgalabel.csv'), index_col=0, header=0)
    tcga_features = torch.from_numpy(tcga_features_df.values).type(torch.float32).to(device)
    tcga_label = torch.from_numpy(tcga_label_df.values).type(torch.float32).squeeze().to(device)

    return (tcga_features, tcga_label)


def TCGA_data_generator(drug):
    tcga_data = TCGA_target_data(drug)
    ccle_data_tuple = TCGA_source_5fold(drug)
    for ccle_train_data, ccle_eval_data in ccle_data_tuple:
        yield (ccle_train_data, ccle_eval_data, tcga_data)