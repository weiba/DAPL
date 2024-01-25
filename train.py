import os
import torch
import math
import copy
import itertools
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.autograd as autograd
from data import *
from copy import deepcopy
from itertools import chain
from collections import defaultdict
from precontext import drug_pretrain
from tools.dataprocess import *
from torch_geometric import data as DATA
from tools.model import *
from drugmodels.ginconv import GINConvNet
from sklearn.metrics import accuracy_score, f1_score, auc, precision_recall_curve, average_precision_score, roc_auc_score
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('use device:', device)

def safemakedirs(folder):
    if os.path.exists(folder) == False:
        os.makedirs(folder)

def ortho_loss(shared_z, private_z):
    s_l2_norm = torch.norm(shared_z, p=2, dim=1, keepdim=True).detach()
    s_l2 = shared_z.div(s_l2_norm.expand_as(shared_z) + 1e-6)
    p_l2_norm = torch.norm(private_z, p=2, dim=1, keepdim=True).detach()
    p_l2 = private_z.div(p_l2_norm.expand_as(private_z) + 1e-6)
    ortho_loss = torch.mean((s_l2.t().mm(p_l2)).pow(2))
    return ortho_loss

def step_drug_train():
    drug_encoder_dict_pth = 'drug_encoder.pth'
    if os.path.exists(drug_encoder_dict_pth):
        print("pretrain done")
        drug_encoder_dict = torch.load(drug_encoder_dict_pth)
        return drug_encoder_dict
    else:
        drug_smiles_df = pd.read_csv(os.path.join('data','smile_inchi326.csv'), index_col=0)
        smiles228 = drug_smiles_df['smiles'].tolist()
        drug_pyg_list = []
        for smile in smiles228:
            _, x, edge_index = smile_to_graph(smile)
            x = torch.tensor(np.array(x), device=device).float()
            edge_index = torch.tensor(edge_index, device=device).t()
            temp_pyg = DATA.Data(
                x=x,
                edge_index=edge_index
            )
            drug_pyg_list.append(temp_pyg)
        drug_encoder_dict = drug_pretrain(drug_pyg_list)
        torch.save(drug_encoder_dict, drug_encoder_dict_pth)
        return drug_encoder_dict

def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.shape[0], 1)).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    # interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples.repeat(real_samples.shape[0] // fake_samples.shape[0], 1))).requires_grad_(True)
    # print("inter shape", interpolates.size())
    critic_interpolates = critic(interpolates)
    fakes = torch.ones((real_samples.shape[0], 1)).to(device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=fakes,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_discrim(s_batch, t_batch, shared_encoder, sencoder,tencoder, discrim, optimizer,scheduler):
    loss_log = defaultdict(float)
    shared_encoder.zero_grad()
    sencoder.zero_grad()
    tencoder.zero_grad()
    discrim.zero_grad()
    sencoder.eval()
    tencoder.eval()
    shared_encoder.eval()
    discrim.train()
    optimizer.zero_grad()

    # pvae
    _,pzs,_,_ = sencoder(s_batch)
    _,pzt,_,_ = tencoder(t_batch)
    # vae
    _,zs,_,_ = shared_encoder(s_batch)
    _,zt,_,_ = shared_encoder(t_batch)
    # F.normalize(latent_code, p=2, dim=1)
    s = torch.cat((zs, pzs), dim=1)
    t = torch.cat((zt, pzt), dim=1)
    d_loss = torch.mean(t)-torch.mean(s)
    g_p = compute_gradient_penalty(
        critic=discrim,
        real_samples=s,
        fake_samples=t,
        device=device
    )
    loss_log.update({"discrim_loss":d_loss, "g_p":g_p})
    d_loss = d_loss + 10*g_p
    d_loss.backward()
    optimizer.step()
    scheduler.step()
    discrim.eval()

    return loss_log

def train_d_ae(s_batch, t_batch, shared_encoder, sencoder, tencoder, discrim,  optimizer,scheduler):
    loss_log = defaultdict(float)
    shared_encoder.zero_grad()
    sencoder.zero_grad()
    tencoder.zero_grad()
    discrim.zero_grad()
    sencoder.train()
    tencoder.train()
    shared_encoder.train()
    discrim.eval()
    optimizer.zero_grad()
    # private
    pccle_re_x, pccle_z, pccle_mu, pccle_sigma = sencoder(s_batch)
    pccle_vae_loss = vaeloss(pccle_mu, pccle_sigma, pccle_re_x, s_batch)
    ptcga_re_x, ptcga_z, ptcga_mu, ptcga_sigma = tencoder(t_batch)
    ptcga_vae_loss = vaeloss(ptcga_mu, ptcga_sigma, ptcga_re_x, t_batch)
    # shared
    ccle_re_x, ccle_z, ccle_mu, ccle_sigma = shared_encoder(s_batch)
    ccle_vae_loss = vaeloss(ccle_mu, ccle_sigma, ccle_re_x, s_batch)
    tcga_re_x, tcga_z, tcga_mu, tcga_sigma = shared_encoder(t_batch)
    tcga_vae_loss = vaeloss(tcga_mu, tcga_sigma, tcga_re_x, t_batch)
    # vaeloss
    pvae_loss = pccle_vae_loss+ptcga_vae_loss
    vae_loss = ccle_vae_loss + tcga_vae_loss
    # oloss
    o_loss = ortho_loss(ccle_z, pccle_z)+ortho_loss(tcga_z,ptcga_z)
    # gloss
    g_loss = -torch.mean(discrim(torch.cat((tcga_z, ptcga_z), dim=1)))
    loss = o_loss + + g_loss + vae_loss + pvae_loss
    loss_log.update({"ortho_loss":o_loss, "pvae_loss":pvae_loss, "gen_loss":g_loss, "vae_loss":vae_loss})
    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss_log

def pretrain(sourcedata, targetdata, param, parent_folder):
    print("start pretrain")
    set_dir_name = 'pt_epochs_' + str(param['pretrain_num_epochs']) + \
                   ',t_epochs_' + str(param['train_num_epochs']) + \
                   ',Ptlr_' + str(param['pretrain_learning_rate']) + \
                   ',tlr' + str(param['gan_learning_rate'])
    pretrain_dir = os.path.join(parent_folder, set_dir_name)
    safemakedirs(pretrain_dir)

    trainloss_logfile = os.path.join(pretrain_dir, "pretrain_losslog.txt")
    evalloss_logfile = os.path.join(pretrain_dir, "pretrain_eval_losslog.txt")

    trainloss_logdict = defaultdict(float)
    evalloss_logdict = defaultdict(float)

    # data
    sourcetrainloader = sourcedata[0]
    sourcetest = sourcedata[1]
    targettrainloader = targetdata[0]
    targettest = targetdata[1]
    # shared
    shared_vae = VAE(input_size=1426, output_size=1426, latent_size=32, hidden_size=128).to(device)
    # private
    source_private_vae = VAE(input_size=1426, output_size=1426, latent_size=32, hidden_size=128).to(device)
    target_private_vae = VAE(input_size=1426, output_size=1426, latent_size=32, hidden_size=128).to(device)
    if os.path.exists(os.path.join(pretrain_dir, "shared_vae.pth")) == False:
        # model dict
        source_dict = source_private_vae.state_dict()
        shared_dict = shared_vae.state_dict()
        target_dict = target_private_vae.state_dict()
        # pretrain parameter
        pretrain_epochs = param['pretrain_num_epochs']
        learning_rate = param['pretrain_learning_rate']
        tolerance = 0
        max_tolerance = 50
        min_loss = float('inf')
        models = [shared_vae, source_private_vae, target_private_vae]
        models_parameters = [
            shared_vae.parameters(),
            source_private_vae.parameters(),
            target_private_vae.parameters()
        ]
        optimizer = torch.optim.Adam(chain(*models_parameters), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, pretrain_epochs)
        for epoch in range(pretrain_epochs):
            if epoch % 20 == 0:
                print("pretrain epoch:", epoch)
            train_epoch_oloss = 0
            train_epoch_vaeloss = 0
            train_epoch_pvaeloss = 0
            # start train
            for model in models:
                model.train()
            # print("train")
            for i, ccledata in enumerate(sourcetrainloader):
                tcgadata = next(iter(targettrainloader))
                optimizer.zero_grad()
                ccledata = ccledata[0]
                tcgadata = tcgadata[0]

                pccle_re_x, pccle_z, pccle_mu, pccle_sigma = source_private_vae(ccledata)
                pccle_vae_loss = vaeloss(pccle_mu, pccle_sigma, pccle_re_x, ccledata)
                ptcga_re_x, ptcga_z, ptcga_mu, ptcga_sigma = target_private_vae(tcgadata)
                ptcga_vae_loss = vaeloss(ptcga_mu, ptcga_sigma, ptcga_re_x, tcgadata)

                ccle_re_x, ccle_z, ccle_mu, ccle_sigma = shared_vae(ccledata)
                ccle_vae_loss = vaeloss(ccle_mu, ccle_sigma, ccle_re_x, ccledata)
                tcga_re_x, tcga_z, tcga_mu, tcga_sigma = shared_vae(tcgadata)
                tcga_vae_loss = vaeloss(tcga_mu, tcga_sigma, tcga_re_x, tcgadata)

                p_vae_loss = pccle_vae_loss + ptcga_vae_loss
                vae_loss = ccle_vae_loss + tcga_vae_loss
                o_loss = ortho_loss(ccle_z, pccle_z) + ortho_loss(tcga_z, ptcga_z)

                loss = o_loss + vae_loss + p_vae_loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_epoch_oloss += o_loss
                train_epoch_vaeloss += vae_loss
                train_epoch_pvaeloss += p_vae_loss

            for model in models:
                model.eval()
            train_epoch_oloss, train_epoch_pvaeloss, train_epoch_vaeloss = train_epoch_oloss / i, train_epoch_pvaeloss / i, train_epoch_vaeloss / i
            trainloss_logdict.update({
                "epoch": epoch,
                "ortholoss": train_epoch_oloss,
                "pVAE_loss": train_epoch_pvaeloss,
                "VAE_loss": train_epoch_vaeloss
            })
            append_file(trainloss_logfile, trainloss_logdict)
            # print("eval")
            with torch.no_grad():

                pccle_re_x, pccle_z, pccle_mu, pccle_sigma = source_private_vae(sourcetest)
                pccle_vae_loss = vaeloss(pccle_mu, pccle_sigma, pccle_re_x, sourcetest)
                ptcga_re_x, ptcga_z, ptcga_mu, ptcga_sigma = target_private_vae(targettest)
                ptcga_vae_loss = vaeloss(ptcga_mu, ptcga_sigma, ptcga_re_x, targettest)
                # vae
                ccle_re_x, ccle_z, ccle_mu, ccle_sigma = shared_vae(sourcetest)
                ccle_vae_loss = vaeloss(ccle_mu, ccle_sigma, ccle_re_x, sourcetest)
                tcga_re_x, tcga_z, tcga_mu, tcga_sigma = shared_vae(targettest)
                tcga_vae_loss = vaeloss(tcga_mu, tcga_sigma, tcga_re_x, targettest)
                # eval vae loss
                eval_vae_loss = ccle_vae_loss + tcga_vae_loss
                eval_pvae_loss = pccle_vae_loss + ptcga_vae_loss
                # eval ortho loss
                eval_oloss = ortho_loss(ccle_z, pccle_z) + ortho_loss(tcga_z, ptcga_z)
                evalloss_logdict.update({
                    "epoch": epoch,
                    "ortholoss": eval_oloss,
                    "pVAE_loss": eval_pvae_loss,
                    "VAE_loss": eval_vae_loss
                })
                append_file(evalloss_logfile, evalloss_logdict)
                # 早停
                evalloss = eval_oloss + eval_pvae_loss + eval_vae_loss
                if evalloss < min_loss:
                    min_loss = evalloss
                    tolerance = 0
                    source_dict = source_private_vae.state_dict()
                    target_dict = target_private_vae.state_dict()
                    shared_dict = shared_vae.state_dict()
                else:
                    tolerance += 1
                if tolerance >= max_tolerance:
                    print("pretrain early stop")
                    break
        torch.save(shared_dict, os.path.join(pretrain_dir, "shared_vae.pth"))
        torch.save(source_dict, os.path.join(pretrain_dir, "source_vae.pth"))
        torch.save(target_dict, os.path.join(pretrain_dir, "target_vae.pth"))
    else:
        shared_dict = torch.load(os.path.join(pretrain_dir, "shared_vae.pth"))
        source_dict = torch.load(os.path.join(pretrain_dir, "source_vae.pth"))
        target_dict = torch.load(os.path.join(pretrain_dir, "target_vae.pth"))
    # train gan
    if os.path.exists(os.path.join(pretrain_dir, 'after_traingan_shared_vae.pth')) == True:
        print("after train gan model exists")
    elif os.path.exists(os.path.join(pretrain_dir, 'after_traingan_shared_vae.pth')) == False:
        print("start gan train")
        gan_epoch = param['train_num_epochs']
        gan_lr = param['gan_learning_rate']
        shared_vae.load_state_dict(shared_dict)
        source_private_vae.load_state_dict(source_dict)
        target_private_vae.load_state_dict(target_dict)
        d_ae_parameters = [
            shared_vae.parameters(),
            source_private_vae.parameters(),
            target_private_vae.parameters()
        ]
        discrim = Discriminator(input_dim=32 + 32).to(device)
        discrim_optimizer = torch.optim.RMSprop(discrim.parameters(), lr=gan_lr)
        discrim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(discrim_optimizer, gan_epoch)
        d_ae_optimizer = torch.optim.RMSprop(chain(*d_ae_parameters), lr=gan_lr)
        d_ae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(d_ae_optimizer, gan_epoch)
        dloss_logfile = os.path.join(pretrain_dir, "discrim_losslog.txt")
        genloss_logfile = os.path.join(pretrain_dir, "generation_losslog.txt")
        max_gan_tolerance = 20
        gan_tolerance = 0
        shared_vae_aftergan_dict = shared_vae.state_dict()
        min_loss = float('inf')
        for epoch in range(gan_epoch):
            temp_loss = 0
            dloss_list = []
            genloss_list = []
            if epoch % 10 == 0:
                print(f'confounder wgan training epoch {epoch}')
            for step, ccledata in enumerate(sourcetrainloader):
                tcgadata = next(iter(targettrainloader))
                dlosslog = train_discrim(s_batch=ccledata[0],
                                         t_batch=tcgadata[0],
                                         shared_encoder=shared_vae,
                                         sencoder=source_private_vae,
                                         tencoder=target_private_vae,
                                         discrim=discrim,
                                         optimizer=discrim_optimizer,
                                         scheduler=discrim_scheduler
                                         )
                dloss_list.append(dlosslog)
                if (step + 1) % 5 == 0:
                    genlosslog = train_d_ae(s_batch=ccledata[0],
                                            t_batch=tcgadata[0],
                                            shared_encoder=shared_vae,
                                            sencoder=source_private_vae,
                                            tencoder=target_private_vae,
                                            discrim=discrim,
                                            optimizer=d_ae_optimizer,
                                            scheduler=d_ae_scheduler
                                            )
                    genloss_list.append(genlosslog)
            # dloss
            dloss_sum = defaultdict(float)
            dloss_mean = defaultdict(float)
            for metric_dict in dloss_list:
                for metric, value in metric_dict.items():
                    dloss_sum[metric] += value
            num_dicts = len(dloss_list)
            for metric, total_value in dloss_sum.items():
                average_value = total_value / num_dicts
                dloss_mean[metric] = average_value
            # genloss
            genloss_sum = defaultdict(float)
            genloss_mean = defaultdict(float)
            for metric_dict in genloss_list:
                for metric, value in metric_dict.items():
                    genloss_sum[metric] += value
            num_dicts = len(genloss_list)
            for metric, total_value in genloss_sum.items():
                average_value = total_value / num_dicts
                genloss_mean[metric] = average_value
            append_file(dloss_logfile, dloss_mean)
            append_file(genloss_logfile, genloss_mean)
            for key in dloss_mean:
                temp_loss = temp_loss + dloss_mean[key].item()
            for key in genloss_mean:
                temp_loss = temp_loss + genloss_mean[key].item()
            if min_loss > temp_loss:
                gan_tolerance = 0
                shared_vae_aftergan_dict = shared_vae.state_dict()
                min_loss = temp_loss
            else:
                gan_tolerance += 1
            if gan_tolerance >= max_gan_tolerance:
                print("train gan early stop in epoch:", epoch)
                break
        torch.save(shared_vae_aftergan_dict, os.path.join(pretrain_dir, "after_traingan_shared_vae.pth"))

def step_pretrain(folder):
    sourcepretrain, targetpretrain = pretrain_data()
    params_grid = {
        "pretrain_num_epochs": [0, 100, 300],
        'pretrain_learning_rate': [0.001],
        'gan_learning_rate': [0.001],
        "train_num_epochs": [100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000]
    }
    keys, values = zip(*params_grid.items())
    update_params_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    for param_dict in update_params_dict_list:
        pretrain(sourcepretrain, targetpretrain, param=param_dict, parent_folder=folder)


def fine_tune(Data, encoder, classifymodel, drugmodel, optimizer, scheduler, drug_data, drug_folder, kfold_num, param1,
              param2):
    # def fine_tune(Data, encoder, classifymodel, optimizer, scheduler, drug_data, drug_folder, kfold_num, param1, param2):
    classification_loss = nn.BCEWithLogitsLoss()
    num_epoch = param2['train_num_epochs']
    best_metrics = {
        'EPOCH': 0,
        'AUC': 0,
        'AUPRC': 0,
        'F1': 0,
        'Accuracy': 0
    }
    best_test_metrics = {
        'EPOCH': 0,
        'AUC': 0,
        'AUPRC': 0,
        'F1': 0,
        'Accuracy': 0
    }
    source_train_data, source_test_data, target_data = Data[0], Data[1], Data[2]
    encoder.eval()
    # encoder.train()
    # protomodel.eval()
    classifymodel.train()
    # drugmodel.train()
    drugmodel.eval()
    # log
    loss_log_name = os.path.join(drug_folder, str(kfold_num) + 'train_loss_log.txt')
    eval_log_name = os.path.join(drug_folder, str(kfold_num) + '_foldeval.txt')
    test_log_name = os.path.join(drug_folder, str(kfold_num) + '_foldtest.txt')
    best_eval_log_name = os.path.join(drug_folder, str(kfold_num) + "_fold_best_auc.txt")

    train_feature_name = os.path.join(drug_folder, str(kfold_num) + '_fold_trainfeature')
    eval_feature_name = os.path.join(drug_folder, str(kfold_num) + '_fold_evalfeature')
    test_feature_name = os.path.join(drug_folder, str(kfold_num) + '_fold_testfeature')
    # # stop
    tolerance = 0
    max_tolerance = 20
    for epoch in range(num_epoch):
        finetune_lossdict = defaultdict(float)
        optimizer.zero_grad()
        # print("train epoch:", epoch)
        _, z, _, _ = encoder(source_train_data[0])
        drugemb = drugmodel(drug_data)
        scatemb = cat_tensor_with_drug(z, drugemb)
        # scatemb = z
        predict = classifymodel(scatemb)
        c_loss = classification_loss(predict, source_train_data[1])
        loss = c_loss
        loss.backward()
        optimizer.step()
        if param1['scheduler_flag'] == True:
            scheduler.step()
        if torch.is_tensor(c_loss):
            c_loss = c_loss.item()
        finetune_lossdict.update({'class_loss': c_loss})
        append_file(loss_log_name, finetune_lossdict)
        # eval and test
        with torch.no_grad():
            _, evalemb, _, _ = encoder(source_test_data[0])
            _, testemb, _, _ = encoder(target_data[0])

            catemb_eval = cat_tensor_with_drug(evalemb, drugemb)
            catemb_test = cat_tensor_with_drug(testemb, drugemb)
            # catemb_eval = evalemb
            # catemb_test = testemb
            _, trainemb, _, _ = encoder(source_train_data[0])
            catemb_train = cat_tensor_with_drug(trainemb, drugemb)

            test_y_pred = classifymodel(catemb_test).cpu().detach().numpy()
            eval_y_pred = classifymodel(catemb_eval).cpu().detach().numpy()
            eval_y_true = source_test_data[1].cpu().detach().numpy()
            test_y_true = target_data[1].cpu().detach().numpy()
            # metrics
            eval_auc = roc_auc_score(eval_y_true, eval_y_pred)
            eval_auprc = average_precision_score(eval_y_true,eval_y_pred)
            # eval_auprc = auprc(eval_y_true, eval_y_pred)
            eval_f1 = f1_score(eval_y_true, (eval_y_pred > 0.5).astype('int'))
            eval_acc = accuracy_score(eval_y_true, (eval_y_pred > 0.5).astype('int'))
            eval_metrics = {
                'EPOCH:': epoch,
                'AUC': eval_auc,
                'AUPRC': eval_auprc,
                'F1': eval_f1,
                'Accuracy': eval_acc
            }
            append_file(eval_log_name, eval_metrics)
            test_auc = roc_auc_score(test_y_true, test_y_pred)
            test_auprc = average_precision_score(test_y_true,test_y_pred)
            # test_auprc = auprc(test_y_true, test_y_pred)
            test_f1 = f1_score(test_y_true, (test_y_pred > 0.5).astype('int'))
            test_acc = accuracy_score(test_y_true, (test_y_pred > 0.5).astype('int'))
            test_metrics = {
                'EPOCH:': epoch,
                'AUC': test_auc,
                'AUPRC': test_auprc,
                'F1': test_f1,
                'Accuracy': test_acc
            }
            append_file(test_log_name, test_metrics)
            # early stop
            if eval_metrics['AUC'] >= best_metrics['AUC']:
                best_metrics.update(eval_metrics)
                best_metrics['EPOCH'] = epoch
                best_test_metrics.update(test_metrics)
                best_test_metrics['EPOCH'] = epoch
                temp_log = {'epoch': epoch, "eval auc=": eval_metrics['AUC'], "test auc=": test_metrics['AUC']}
                append_file(best_eval_log_name, temp_log)
                tolerance = 0
                best_train_feature = catemb_train
                best_eval_feature = catemb_eval
                best_test_feature = catemb_test
                # pretrainc1
                best_classifier = copy.deepcopy(classifymodel)
            else:
                tolerance += 1
                if tolerance in (10, 20, 50):
                    append_file(best_eval_log_name, {'early stop': tolerance})
            if tolerance >= max_tolerance:
                # print("Early stopping triggered. Training stopped.")
                # best_eval_log_name = os.path.join(drug_folder, "best_auc.txt")
                # append_file(best_eval_log_name, best_metrics)
                break
    # train_feature_name = os.path.join(drug_folder, str(kfold_num)+'_fold_trainfeature')
    # eval_feature_name = os.path.join(drug_folder, str(kfold_num)+'_fold_evalfeature')
    # test_feature_name = os.path.join(drug_folder, str(kfold_num)+'_fold_testfeature')
    torch.save(best_train_feature, train_feature_name)
    torch.save(best_eval_feature, eval_feature_name)
    torch.save(best_test_feature, test_feature_name)
    torch.save(best_classifier.state_dict(), os.path.join(drug_folder, str(kfold_num) + 'fold_classifier.pth'))
    print('{}_fold feature and classifier saved , best_test_auc:{}'.format(kfold_num, best_test_metrics['AUC']))
    return best_test_metrics

def step_1_finetune(parent_folder, drug_list, drug_smiles, datatype):
    drug_encoder_dict = step_drug_train()
    params_grid = {
        "pretrain_num_epochs": [0, 100, 300],
        'pretrain_learning_rate': [0.001],
        'gan_learning_rate': [0.001],
        "train_num_epochs": [100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000]
    }
    keys, values = zip(*params_grid.items())
    update_params_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    fine_tune_params_grid = {
        'ftlr' : [0.01, 0.001],
        'scheduler_flag' : [True, False]
    }
    ftkeys, ftvalues = zip(*fine_tune_params_grid.items())
    fine_tune_dict_list = [dict(zip(ftkeys, v)) for v in itertools.product(*ftvalues)]
    all_metrics =  {}
    for drug in drug_list:
        all_metrics.update({drug:0, drug+'folder':None})
    for fine_tune_dict in fine_tune_dict_list:
        for param in update_params_dict_list:
            for drug, drug_smile in zip(drug_list, drug_smiles):
                set_dir_name = 'pt_epochs_' + str(param['pretrain_num_epochs']) + \
                               ',t_epochs_' + str(param['train_num_epochs']) + \
                               ',Ptlr_' + str(param['pretrain_learning_rate']) + \
                               ',tlr' + str(param['gan_learning_rate'])
                model_folder = os.path.join(parent_folder, set_dir_name)
                encoder_state_dict = torch.load(os.path.join(model_folder, 'after_traingan_shared_vae.pth'))
                print('train drug:', drug)
                # Drug Data : DATA.Data
                _, x, edge_index = smile_to_graph(drug_smile)
                x = torch.tensor(np.array(x), device=device).float()
                edge_index = torch.tensor(edge_index, device=device).t()
                drug_data = DATA.Data(
                    x=x,
                    edge_index=edge_index
                )
                if datatype == 'PDTC':
                    data_generator = PDTC_data_generator(drug)
                elif datatype == 'TCGA':
                    data_generator = TCGA_data_generator(drug)
                auc_folder = os.path.join(model_folder, 'feature_save')
                drug_auc_folder = os.path.join(auc_folder, drug)
                safemakedirs(drug_auc_folder)
                test_auc_list = []
                i = 0  # fold num
                addauc = 0
                for data in data_generator:
                    temp_folder = os.path.join(drug_auc_folder,"ftepoch"+str(param['train_num_epochs'])+",lr:"+str(fine_tune_dict['ftlr'])+",CosAL:"+str(fine_tune_dict['scheduler_flag']))
                    log_folder = os.path.join(temp_folder, 'log')
                    safemakedirs(log_folder)
                    test_auc_log_name = os.path.join(temp_folder, 'step1_test_auc.txt')
                    temp_encoder = VAE(input_size=1426, output_size=1426, latent_size=32, hidden_size=128).to(device)
                    temp_encoder.load_state_dict(encoder_state_dict)
                    classifymodel = Classify(input_dim=32+10).to(device)
                    drug_gcnmodel = GINConvNet(input_dim=drug_data.x.shape[1], output_dim=10).to(device)
                    drug_gcnmodel.load_state_dict(drug_encoder_dict)
                    fine_tune_optimizer = torch.optim.AdamW(classifymodel.parameters(), lr=fine_tune_dict['ftlr'])
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fine_tune_optimizer, param['train_num_epochs'])
                    test_history = fine_tune(
                        Data = data,
                        encoder=temp_encoder,
                        drugmodel=drug_gcnmodel,
                        classifymodel=classifymodel,
                        optimizer = fine_tune_optimizer,
                        scheduler = scheduler,
                        drug_data=drug_data,
                        drug_folder=log_folder,
                        kfold_num = i,
                        param1 = fine_tune_dict,
                        param2 = param
                    )
                    test_auc_list.append(test_history)
                    addauc += test_history['AUC']
                    i=i+1
                    if i==5:
                        meanauc = addauc/5
                        if meanauc > all_metrics[drug]:
                            all_metrics[drug] = meanauc
                            all_metrics[drug+'folder'] = temp_folder
                        print('pretrain mean auc:', addauc/5)
                        with open(test_auc_log_name,'w') as f:
                            for item in test_auc_list:
                                f.write(str(item)+'\n')
                        f.close()
    return all_metrics

def label1_2(tensor):
    return torch.cat([1-tensor.view(-1,1), tensor.view(-1,1)],dim=1)

@torch.no_grad()
def m_encoder_update(encoder, m_encoder, m=0.9):
    """
    update momentum encoder
    """
    moco_m = m # 0.999
    for param_q, param_k in zip(encoder.parameters(), m_encoder.parameters()):
        param_k.data = param_k.data * moco_m + param_q.data * (1. - moco_m)
    return

def p_step2(Data, c2, encoder, decoder):
    # data
    source_train_data, source_test_data, target_data = Data[0], Data[1], Data[2]
    step1_train_models = [c2, encoder, decoder]
    step1_parameters = []
    for m in step1_train_models:
        m.train()
        step1_parameters.append(m.parameters())
    # loss
    reloss = nn.MSELoss()
    closs = nn.CrossEntropyLoss()
    # parameters
    optimizer = optim.AdamW(chain(*step1_parameters), lr=0.001)
    num_epochs = 1000
    # early stop
    tolerance = 0
    max_tolerance = 20
    best_eval_auc = 0
    for epoch in range(num_epochs):
        # 42dim
        ccle_trainemb = source_train_data[0]
        # 42->10 encoder 10->42 decoder
        ccle_lowemb = encoder(ccle_trainemb)
        ccle_trainemb_re = decoder(ccle_lowemb)
        recon_loss = reloss(ccle_trainemb_re, ccle_trainemb)
        # c2 dim2 classifier
        c2_predict = c2(ccle_lowemb)
        c2_loss = closs(c2_predict, label1_2(source_train_data[1]))
        # loss and backward
        loss = recon_loss+c2_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # early stop(use c2???)
        # eval auc
        ccle_evalemb = source_test_data[0]
        ccle_evallowemb = encoder(ccle_evalemb)
        ccle_eval_predict = c2(ccle_evallowemb) # predict
        ccle_eval_true = label1_2(source_test_data[1]).cpu().detach().numpy()
        epoch_eval_auc = roc_auc_score(ccle_eval_true, ccle_eval_predict.cpu().detach().numpy())
        # early stop
        if epoch_eval_auc>best_eval_auc:
            tolerance=0
            best_eval_auc = epoch_eval_auc
            # best model
            best_encoder = deepcopy(encoder)
            best_decoder = deepcopy(decoder)
            best_c2 = deepcopy(c2)
        else:
            tolerance+=1
        if tolerance>=max_tolerance:
            print('step1 stop at:', epoch)
            break
    return best_encoder, best_decoder, best_c2

def init_prototypes(Data,Mencoder):
    # data
    source_train_data = Data[0]
    cclez = source_train_data[0]
    ccle_compress = Mencoder(cclez)
    label_0_indices = torch.where(source_train_data[1] == 0)[0]
    label_1_indices = torch.where(source_train_data[1] == 1)[0]
    ccle_compress_0 = ccle_compress[label_0_indices]
    ccle_compress_1 = ccle_compress[label_1_indices]
    prototype0 = F.normalize(torch.mean(ccle_compress_0.clone(), dim=0), p=2,dim=0)
    prototype1 = F.normalize(torch.mean(ccle_compress_1.clone(), dim=0), p=2,dim=0)
    prototypes = torch.cat((prototype0.unsqueeze(0), prototype1.unsqueeze(0)), dim=0)
    return prototypes

def step_relationmodel(Data, prototypes, r_model, low_encoder):
    print('train relation model')
    source_train_data, source_test_data, target_data = Data[0], Data[1], Data[2]
    bceloss = nn.BCELoss()
    train_num_epochs = 1000
    low_encoder.eval()
    r_model.train()
    optimizer = optim.AdamW(r_model.parameters(), lr=0.001)
    min_loss = float('inf')
    tolerance = 0
    max_tolerance = 10
    best_r_model = None
    for epoch in range(train_num_epochs):
        label_0_indices = torch.where(source_train_data[1] == 0)[0]
        label_1_indices = torch.where(source_train_data[1] == 1)[0]
        lowtrain = low_encoder(source_train_data[0])
        # cat with prototypes0
        cat_p0_data = cat_tensor_with_drug(lowtrain, prototypes[0,:])
        cat_p0_label = torch.zeros_like(source_train_data[1])
        cat_p0_label[label_0_indices] = 1
        cat_p0_predict = r_model(cat_p0_data)
        cat_p0_loss = bceloss(cat_p0_predict, cat_p0_label)
        # cat with prototypes1
        cat_p1_data = cat_tensor_with_drug(lowtrain, prototypes[1,:])
        cat_p1_label = torch.zeros_like(source_train_data[1])
        cat_p1_label[label_1_indices] = 1
        cat_p1_predict = r_model(cat_p1_data)
        cat_p1_loss = bceloss(cat_p1_predict, cat_p1_label)
        # loss
        loss = cat_p0_loss + cat_p1_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if loss.item()<min_loss:
            min_loss = loss
            tolerance = 0
            best_r_model = deepcopy(r_model)
        else:
            tolerance +=1
        if tolerance>=max_tolerance:
            print('early stop at epoch:', epoch)
            break
    r_model = deepcopy(best_r_model)
    # return r_model

def p_step3(Data, step3optimizer, encoder, Mencoder, epoch, c1, lin, r_model):
    # data
    source_train_data, source_test_data, target_data = Data[0], Data[1], Data[2]
    # step2
    # step2 model setting
    encoder.train()
    lin.train()
    c1.eval()
    Mencoder.eval()
    # loss
    bcelogitsloss = nn.BCEWithLogitsLoss()
    reloss = nn.MSELoss()
    closs = nn.CrossEntropyLoss()
    # 42dim
    ccle_trainemb = source_train_data[0]
    # 42->10 encoder 10->42 decoder
    ccle_lowemb = encoder(ccle_trainemb)
    # prototypes
    source_train_data = Data[0]
    cclez = source_train_data[0]
    ccle_compress = Mencoder(cclez)
    label_0_indices = torch.where(source_train_data[1] == 0)[0]
    label_1_indices = torch.where(source_train_data[1] == 1)[0]
    ccle_compress_0 = ccle_compress[label_0_indices]
    ccle_compress_0 = lin(ccle_compress_0)
    ccle_compress_1 = ccle_compress[label_1_indices]
    ccle_compress_1 = lin(ccle_compress_1)
    mean_ccle_0 = torch.mean(ccle_compress_0.clone(), dim=0)
    mean_ccle_1 = torch.mean(ccle_compress_1.clone(), dim=0)
    prototype0 = mean_ccle_0/torch.norm(mean_ccle_0, p=2)
    prototype1 = mean_ccle_1/torch.norm(mean_ccle_1, p=2)
    prototypes = torch.cat((prototype0.unsqueeze(0), prototype1.unsqueeze(0)), dim=0)
    # contrast
    tcgacat = target_data[0]
    tcga_compress = encoder(tcgacat)
    tcga_compress_m = Mencoder(tcgacat)
    tcga_mm = torch.mm(tcga_compress, tcga_compress_m.t())
    tcga_conlabel = torch.eye(tcga_mm.shape[0], tcga_mm.shape[0]).to(device)
    conloss_tcga = closs(tcga_mm, tcga_conlabel)

    temperature = 0.1
    logits_proto = torch.mm(tcga_compress, prototypes.t())
    logits_proto_raw = logits_proto.detach().clone()
    alpha = 0.5
    # alpha = 0.4+epoch/50*(1-0.4)
    tcga_c1_predict = c1(tcgacat)

    soft_c2predict = torch.zeros(tcga_c1_predict.shape[0], 2).to(device)
    soft_c2predict[:,1] = torch.sigmoid(tcga_c1_predict)
    soft_c2predict[:,0] = 1-torch.sigmoid(tcga_c1_predict)
    unlabel_soft = alpha*soft_c2predict + (1-alpha)*F.softmax(logits_proto_raw, dim=1) # target_soft, [n, 2] 由分类器输出&sample-prototype相似度加权(直接用标签为1的概率)
    yuzhi = 0.5+epoch/100*(1-0.5)
    index1 = torch.where(unlabel_soft[:,1]>yuzhi)[0]
    index0 = torch.where(unlabel_soft[:,0]>yuzhi)[0]

    unlabel_soft_1 = F.softmax(logits_proto_raw, dim=1)
    index1 = list(set(torch.where(unlabel_soft_1[:,1]>yuzhi)[0]) | set(index1))
    index0 = list(set(torch.where(unlabel_soft_1[:,0]>yuzhi)[0]) | set(index0))
    # relation model
    rindex1_predict = r_model(cat_tensor_with_drug(tcga_compress, prototype1))
    rindex1 = torch.where(rindex1_predict>yuzhi)[0]
    rindex0_predict = r_model(cat_tensor_with_drug(tcga_compress, prototype0))
    rindex0 = torch.where(rindex0_predict>yuzhi)[0]

    index1 = torch.tensor(list(set(rindex1) | set(index1)), device=device).long()
    index0 = torch.tensor(list(set(rindex0) | set(index0)), device=device).long()
    correct_idx = torch.cat((index1, index0), dim=0)
    print('step2 correct idx:', correct_idx)

    # distance_sum / (num_visited*torch.log(num_visited+10.)+1e-7)
    fai0_c0 = torch.norm(torch.mean(tcga_compress[index0]-prototypes[0,:].unsqueeze(0), dim=0),p=2)/(len(index0)*math.log(len(index0)+10.)+1e-7)
    fai0_c1 = torch.norm(torch.mean(tcga_compress[index0]-prototypes[1,:].unsqueeze(0), dim=0),p=2)/(len(index0)*math.log(len(index0)+10.)+1e-7)
    fai1_c0 = torch.norm(torch.mean(tcga_compress[index1]-prototypes[0,:].unsqueeze(0), dim=0),p=2)/(len(index1)*math.log(len(index1)+10.)+1e-7)
    fai1_c1 = torch.norm(torch.mean(tcga_compress[index1]-prototypes[1,:].unsqueeze(0), dim=0),p=2)/(len(index1)*math.log(len(index1)+10.)+1e-7)
    posl = torch.cat((torch.mm(tcga_compress[index0], prototypes[0,:].unsqueeze(0).t())/fai0_c0, \
                        torch.mm(tcga_compress[index1], prototypes[1,:].unsqueeze(0).t())/fai1_c1), dim=0)
    negl = torch.cat((torch.mm(tcga_compress[index0], prototypes[1,:].unsqueeze(0).t())/fai0_c1, \
                        torch.mm(tcga_compress[index1], prototypes[0,:].unsqueeze(0).t())/fai1_c0, \
                        torch.mm(prototypes[0,:].unsqueeze(0), prototypes[1,:].unsqueeze(0).t())/temperature), dim=0)
    l = torch.cat((posl, negl), dim=0).to(device)
    lb = torch.cat((torch.ones_like(posl), torch.zeros_like(negl)), dim=0).float().to(device)
    try:
        conloss_unlabeltopro = bcelogitsloss(l, lb)
    except:
        conloss_unlabeltopro = 0
    # loss and backward
    loss = conloss_tcga+conloss_unlabeltopro
    step3optimizer.zero_grad()
    loss.backward(retain_graph=True)
    step3optimizer.step()
    # update m encoder
    m_encoder_update(encoder, Mencoder, m=0.9)
    return loss.item()

def p_step4(Data, step4optimizer, encoder, c1, epoch, mencoder, lin, r_model):
    # data
    source_train_data, source_test_data, target_data = Data[0], Data[1], Data[2]
    # loss
    bcelogitsloss = nn.BCEWithLogitsLoss()
    closs = nn.CrossEntropyLoss()
    # step3
    # step3 model setting
    c1.train()
    encoder.eval()
    lin.train()

    ccle_trainemb = source_train_data[0]
    ccle_c1_predict = c1(ccle_trainemb)

    source_train_data = Data[0]
    cclez = source_train_data[0]
    ccle_compress = mencoder(cclez)
    label_0_indices = torch.where(source_train_data[1] == 0)[0]
    label_1_indices = torch.where(source_train_data[1] == 1)[0]
    ccle_compress_0 = ccle_compress[label_0_indices]
    ccle_compress_1 = ccle_compress[label_1_indices]
    ccle_compress_0 = lin(ccle_compress_0)
    ccle_compress_1 = lin(ccle_compress_1)
    mean_ccle_0 = torch.mean(ccle_compress_0.clone(), dim=0)
    mean_ccle_1 = torch.mean(ccle_compress_1.clone(), dim=0)
    prototype0 = mean_ccle_0/torch.norm(mean_ccle_0, p=2)
    prototype1 = mean_ccle_1/torch.norm(mean_ccle_1, p=2)
    # prototype0 = F.normalize(torch.mean(ccle_compress_0.clone(), dim=0), p=2,dim=0)
    # prototype1 = F.normalize(torch.mean(ccle_compress_1.clone(), dim=0), p=2,dim=0)
    prototypes = torch.cat((prototype0.unsqueeze(0), prototype1.unsqueeze(0)), dim=0)
    # tcga

    tcgacat = target_data[0]
    tcga_c1_predict = c1(tcgacat)
    tcga_compress = encoder(tcgacat)

    alpha=0.5
    # alpha = 0.4+epoch/50*(1-0.4)
    logits_proto = torch.mm(tcga_compress, prototypes.t())
    logits_proto_raw = logits_proto.detach().clone()

    # plabel
    soft_c2predict = torch.zeros(tcga_c1_predict.shape[0], 2).to(device)
    soft_c2predict[:,1] = torch.sigmoid(tcga_c1_predict)
    soft_c2predict[:,0] = 1-torch.sigmoid(tcga_c1_predict)
    unlabel_soft = alpha*soft_c2predict + (1-alpha)*F.softmax(logits_proto_raw, dim=1)

    yuzhi = 0.5+epoch/100*(1-0.5)
    index1 = torch.where(unlabel_soft[:,1]>yuzhi)[0]
    index0 = torch.where(unlabel_soft[:,0]>yuzhi)[0]

    unlabel_soft_1 = F.softmax(logits_proto_raw, dim=1)
    index1 = list(set(torch.where(unlabel_soft_1[:,1]>yuzhi)[0]) | set(index1))
    index0 = list(set(torch.where(unlabel_soft_1[:,0]>yuzhi)[0]) | set(index0))
    # relation model
    rindex1_predict = r_model(cat_tensor_with_drug(tcga_compress, prototype1))
    rindex1 = torch.where(rindex1_predict>yuzhi)[0]
    rindex0_predict = r_model(cat_tensor_with_drug(tcga_compress, prototype0))
    rindex0 = torch.where(rindex0_predict>yuzhi)[0]

    index1 = torch.tensor(list(set(rindex1) | set(index1)), device=device).long()
    index0 = torch.tensor(list(set(rindex0) | set(index0)), device=device).long()
    correct_idx = torch.cat((index1, index0), dim=0)
    print('step3 correct idx:', correct_idx)
    # loss
    loss_cls_hard_ccle = closs(ccle_c1_predict, source_train_data[1])
    if len(correct_idx) != 0:
        # p_label = torch.zeros(index1.shape[0]+index0.shape[0], 2).to(device)
        # p_label[:index1.shape[0], 1] = 1
        # p_label[index1.shape[0]:, 0] = 1
        p_label = torch.zeros(index1.shape[0]+index0.shape[0]).to(device)
        p_label[:index1.shape[0]] = 1
        # loss_cls_soft_unlabel = -torch.sum(p_label * torch.log(F.softmax(tcga_predict[correct_idx], dim=1)))
        loss_cls_soft_unlabel = bcelogitsloss(tcga_c1_predict[correct_idx], p_label)
        loss_cls = loss_cls_hard_ccle + loss_cls_soft_unlabel
    else:
        loss_cls = loss_cls_hard_ccle
    step4optimizer.zero_grad()
    loss_cls.backward()
    step4optimizer.step()

    ccle_compress = mencoder(ccle_trainemb)
    label_0_indices = torch.where(source_train_data[1] == 0)[0]
    label_1_indices = torch.where(source_train_data[1] == 1)[0]
    ccle_compress_0 = ccle_compress[label_0_indices]
    ccle_compress_1 = ccle_compress[label_1_indices]
    prototype0 = F.normalize(torch.mean(ccle_compress_0.clone(), dim=0), p=2,dim=0)
    prototype1 = F.normalize(torch.mean(ccle_compress_1.clone(), dim=0), p=2,dim=0)
    prototypes = torch.cat((prototype0.unsqueeze(0), prototype1.unsqueeze(0)), dim=0)
    # with torch.no_grad():
    #     if len(correct_idx) != 0:
    #         if len(index0) != 0:
    #             prototypes0 = 0.9*prototypes[0,:] + 0.1*torch.mean(tcga_compress[index0].clone(), dim=0)
    #         else:
    #             prototypes0 = prototypes[0,:]
    #         if len(index1) != 0:
    #             prototypes1 = 0.9*prototypes[1,:] + 0.1*torch.mean(tcga_compress[index1].clone(), dim=0)
    #         else:
    #             prototypes1 = prototypes[1,:]
    #         prototypes = torch.cat((prototypes0.unsqueeze(0), prototypes1.unsqueeze(0)),dim=0)
    return correct_idx

def p_fine_tune(Data, classifymodel, kfold_num, folder):
    source_train_data, source_test_data, target_data = Data[0], Data[1], Data[2]
    kfold_model_folder = os.path.join(folder, str(kfold_num)+'model_save')
    if os.path.exists(kfold_model_folder) == False:
        os.mkdir(kfold_model_folder)
    # start proto learning
    lowencoder = projector(in_dim=42, out_dim=10).to(device) # encoder
    lowdecoder = projector_decoder(in_dim=10, out_dim=42).to(device) # decoder
    auxiliary_classifier = Classifydim2(input_dim=10).to(device) # auxiliary classifier
    # step , init lowe lowde aux and M_low
    lowencoder,lowdecoder,auxiliary_classifier = p_step2(Data=Data,
                                                         c2 = auxiliary_classifier,
                                                         encoder = lowencoder,
                                                         decoder = lowdecoder)
    # init M param
    M_lowencoder = projector(in_dim=42, out_dim=10).to(device) # M encoder
    for param_q, param_k in zip(lowencoder.parameters(), M_lowencoder.parameters()):
        param_k.data.copy_(param_q.data)  # initialize
        param_k.requires_grad = False
    # init prototypes
    prototypes = init_prototypes(Data=Data, Mencoder=M_lowencoder)
    # train relation model
    r_model = relation_model(indim=20).to(device)
    step_relationmodel(Data=Data, prototypes=prototypes, r_model=r_model, low_encoder=lowencoder)
    # step2 and step3
    step3_4epochs = 100 #
    lin = nn.Linear(10, 10).to(device)
    step3_optimizer = optim.AdamW(chain(*[lowencoder.parameters(), lin.parameters(), r_model.parameters()]),lr=0.0001)
    step4_optimizer = optim.AdamW(chain(*[lin.parameters(), classifymodel.parameters(), r_model.parameters()]), lr=0.0001)
    best_eval_metrics2 = {'EPOCH': 0,'AUC': 0,'AUPRC': 0,'Accuracy': 0}
    # best_test_metrics2 = {'EPOCH': 0,'AUC': 0,'AUPRC': 0,'Accuracy': 0}
    tolerance = 0
    # max_tolerance= 20
    step2_tolerance = 0
    step2_max_tolerance = 20
    minstep3loss = float('inf')
    best_lowencoder = None
    best_M_encoder = None
    best_lin = None
    best_r_model = None
    for epoch in range(step3_4epochs):
        step3loss = p_step3(Data, step3_optimizer, lowencoder, M_lowencoder, epoch, classifymodel, lin, r_model)
        if minstep3loss > step3loss:
            minstep3loss = step3loss
            best_lowencoder = deepcopy(lowencoder)
            best_M_encoder = deepcopy(M_lowencoder)
            best_lin = lin
            best_r_model = r_model
            step2_tolerance = 0
        else:
            step2_tolerance += 1
        if step2_tolerance >= step2_max_tolerance:
            break
    best_c1_afterp = None
    result_prediction = None
    for epoch in range(step3_4epochs):
        idx = p_step4(Data, step4_optimizer, best_lowencoder, classifymodel, epoch, best_M_encoder, best_lin, best_r_model)
        with torch.no_grad():
            tcgacat = target_data[0]
            ccleevalcat = source_test_data[0]
            # predict
            # test_y_pred = classifymodel(tcgacat).cpu().detach().numpy()
            test_y_pred = torch.sigmoid(test_y_pred)
            eval_y_pred = classifymodel(ccleevalcat).cpu().detach().numpy()
            # eval_y_pred = torch.sigmoid(eval_y_pred)
            eval_y_true = source_test_data[1].cpu().detach().numpy()
            # test_y_true = target_data[1].cpu().detach().numpy()
            # metrics eval
            eval_auc = roc_auc_score(eval_y_true, eval_y_pred)
            eval_auprc = average_precision_score(eval_y_true, eval_y_pred)
            eval_acc = accuracy_score(eval_y_true, (eval_y_pred>0.5).astype('int'))
            eval_metrics = {'EPOCH': epoch,'AUC': eval_auc,'AUPRC': eval_auprc,'Accuracy': eval_acc}
            # metrics test
            # test_auc = roc_auc_score(test_y_true, test_y_pred)
            # test_auprc = average_precision_score(test_y_true,test_y_pred)
            # test_acc = accuracy_score(test_y_true, (test_y_pred>0.5).astype('int'))
            # test_metrics = {'EPOCH': epoch,'AUC': test_auc,'AUPRC': test_auprc,'Accuracy': test_acc}
            if eval_metrics['AUC'] >= best_eval_metrics2['AUC']:
                result_prediction = test_y_pred
                # best_eval_metrics2.update(eval_metrics)
                # best_eval_metrics2['EPOCH'] = epoch
                # best_test_metrics2.update(test_metrics)
                # best_test_metrics2['EPOCH'] = epoch
                # temp_log = {'epoch':epoch,"eval auc=":eval_metrics['AUC'],"test auc=":test_metrics['AUC']}
                tolerance = 0
                # best model
                best_c1_afterp = classifymodel
                # best_c2_afterp = auxiliary_classifier
            else:
                tolerance += 1
        if len(idx) == 0:
            break

    torch.save(best_c1_afterp.state_dict(), os.path.join(kfold_model_folder, 'p_classifier.pth'))
    return result_prediction

def prototype_learning(drug_list, step1_metrics, datatype):
    # resultdf = pd.DataFrame(index=drug_list, columns=['auc', 'aucvar', 'aupr', 'auprvar'])
    for drug in drug_list:
        result_p_path = os.path.join('result_folder', 'presult_'+drug)
        safemakedirs(result_p_path)
        print('train drug:', drug)
        drugauc = 0
        drugaupr = 0
        drug_auc_list = []
        drugaupr_list = []
        if datatype == 'PDTC':
            data_generator = PDTC_data_generator(drug)
        elif datatype == 'TCGA':
            data_generator = TCGA_data_generator(drug)
        test_history_list = []
        i = 0 # fold num
        for data in data_generator:
            temp_folder = step1_metrics[drug+'folder']
            proto_folder = os.path.join(temp_folder, 'proto_model')
            safemakedirs(proto_folder)
            test_auc_log_name = os.path.join(temp_folder, 'proto_test_auc.txt')
            print("fold:", i)
            classifymodel = Classify(input_dim=32+10).to(device)
            classifymodel_dict = torch.load(os.path.join(temp_folder, 'log', str(i)+'fold_classifier.pth'))
            classifymodel.load_state_dict(classifymodel_dict)
            train_feature = torch.load(os.path.join(temp_folder, 'log', str(i)+'_fold_trainfeature'))
            eval_feature = torch.load(os.path.join(temp_folder, 'log', str(i)+'_fold_evalfeature'))
            test_feature = torch.load(os.path.join(temp_folder, 'log', str(i)+'_fold_testfeature'))
            source_train_data, source_test_data, target_data = data[0], data[1], data[2]
            new_data = (train_feature, source_train_data[1]),(eval_feature, source_test_data[1]),(test_feature, target_data[1])
            result_prediction = p_fine_tune(
                Data = new_data,
                classifymodel = classifymodel,
                kfold_num = i,
                folder = proto_folder
            )
            # test_history_list.append(test_history)
            # drugauc += test_history['AUC']
            # drug_auc_list.append(test_history['AUC'])
            # drugaupr += test_history['AUPRC']
            # drugaupr_list.append(test_history['AUPRC'])
            i=i+1
            np.save(os.path.join(result_p_path, str(i)+'_fold.npy'), result_prediction)
            # if i==5:
            #     resultdf.at[drug, 'auc'] = sum(drug_auc_list)/len(drug_auc_list)
            #     resultdf.at[drug, 'aupr'] = sum(drugaupr_list)/len(drugaupr_list)
            #     resultdf.at[drug, 'aucvar'] = np.var(drug_auc_list)
            #     resultdf.at[drug, 'auprvar'] = np.var(drugaupr_list)
            #     with open(test_auc_log_name,'w') as f1:
            #         for item in test_history_list:
            #             f1.write(str(item)+'\n')
            #     f1.close()
    # print(presult)
    # result_path = os.path.join('result' , 'p_result.csv')
    # safemakedirs(result_path)
    # resultdf.to_csv(result_path, index=True)



def step_prototype_learning(parent_folder, drug_list, drug_smiles, datatype):
    step1_metrics = step_1_finetune(parent_folder, drug_list, drug_smiles, datatype)
    prototype_learning(drug_list, step1_metrics, datatype)


def main(drug_list, drug_smiles, datatype):
    parent_folder = os.path.join("result_folder", datatype)
    safemakedirs(parent_folder)
    step_pretrain(folder=parent_folder)
    _ = step_drug_train()
    step_prototype_learning(parent_folder, drug_list, drug_smiles, datatype)

if __name__ == '__main__':
    for datatype in ['TCGA', 'PDTC', 'TCGA']:
        if datatype == 'PDTC':
            pdtc_drug_file = pd.read_csv(os.path.join('data', 'pdtc_gdsc_drug_mapping.csv'), index_col=0)
            drug_list = pdtc_drug_file.index.tolist()
            drug_smiles = pdtc_drug_file['smiles'].tolist()
        elif datatype == 'TCGA':
            drug_list = ['cis', 'sor', 'tem', 'gem', 'fu']
            drug_smiles = ['N.N.Cl[Pt]Cl', 'CNC(=O)C1=NC=CC(=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F',
                           'CN1C(=O)N2C=NC(=C2N=N1)C(=O)N', 'C1=CN(C(=O)N=C1N)C2C(C(C(O2)CO)O)(F)F', 'C1=C(C(=O)NC(=O)N1)F']
        main(drug_list, drug_smiles, datatype)