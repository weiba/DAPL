import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

drug_list_TCGA = ['cis', 'sor', 'tem', 'gem', 'fu']
pdtc_drug_file = pd.read_csv(os.path.join('data', 'pdtc_gdsc_drug_mapping.csv'), index_col=0)
drug_list_PDTC = pdtc_drug_file.index.tolist()

for drug in drug_list_TCGA:
    resultdf = pd.DataFrame(index=drug_list_TCGA, columns=['auc', 'aucvar', 'aupr', 'auprvar'])
    try:
        test_auc_list = []
        test_auprc_list = []
        label_df = pd.read_csv(os.path.join('data', 'TCGA', drug + 'data', 'tcgalabel.csv'), index_col=0, header=0)
        test_y_true = label_df.values.squeeze()
        result_p_path = os.path.join('./result/prototypical', drug)
        for i in range(5):
            test_y_pred = np.load(os.path.join(result_p_path, 'prediction', drug+'_'+str(i)+'_fold.npy'))
            test_auc = roc_auc_score(test_y_true, test_y_pred)
            test_auprc = average_precision_score(test_y_true, test_y_pred)
            test_auc_list.append(test_auc)
            test_auprc_list.append(test_auprc)
        print('drug:{}, 5foldmeanauc:{}, 5foldmeanauprc:{}'.format(drug, sum(test_auc_list)/len(test_auc_list), sum(test_auprc_list)/len(test_auprc_list)))
        resultdf.at[drug, 'auc'] = sum(test_auc_list)/len(test_auc_list)
        resultdf.at[drug, 'aupr'] = sum(test_auprc_list)/len(test_auprc_list)
        resultdf.at[drug, 'aucvar'] = np.var(test_auc_list)
        resultdf.at[drug, 'auprvar'] = np.var(test_auprc_list)
        resultdf.to_csv(os.path.join('result_folder', 'resulttcga.csv'))
    except:
        print('{} presult dont exists'.format(drug))


for drug in drug_list_PDTC:
    resultdf = pd.DataFrame(index=drug_list_TCGA, columns=['auc', 'aucvar', 'aupr', 'auprvar'])
    try:
        test_auc_list = []
        test_auprc_list = []
        label_df = pd.read_csv(os.path.join('data', 'PDTC', drug + 'data', 'pdtclabel.csv'), index_col=0, header=0)
        test_y_true = label_df.values.squeeze()
        result_p_path = os.path.join('./result/prototypical', drug)
        for i in range(5):
            test_y_pred = np.load(os.path.join(result_p_path, 'prediction', drug+'_'+str(i)+'_fold.npy'))
            test_auc = roc_auc_score(test_y_true, test_y_pred)
            test_auprc = average_precision_score(test_y_true, test_y_pred)
            test_auc_list.append(test_auc)
            test_auprc_list.append(test_auprc)
        print('drug:{}, 5foldmeanauc:{}, 5foldmeanauprc:{}'.format(drug, sum(test_auc_list)/len(test_auc_list), sum(test_auprc_list)/len(test_auprc_list)))
        resultdf.at[drug, 'auc'] = sum(test_auc_list)/len(test_auc_list)
        resultdf.at[drug, 'aupr'] = sum(test_auprc_list)/len(test_auprc_list)
        resultdf.at[drug, 'aucvar'] = np.var(test_auc_list)
        resultdf.at[drug, 'auprvar'] = np.var(test_auprc_list)
        resultdf.to_csv(os.path.join('result_folder', 'resultpdtc.csv'))
    except:
        print('{} presult dont exists'.format(drug))