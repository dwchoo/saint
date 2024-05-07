import torch
from torch import nn
from models import SAINT, SAINT_vision

from data_openml import data_prep_openml,task_dset_ids,DataSetCatCon
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import count_parameters, classification_scores, mean_sq_error
from augmentations import embed_data_mask
from augmentations import add_noise

from utils import CosineAnnealingWarmUpRestarts, EarlyStopping, infinite_pop

import os
import numpy as np
import time
import random
from datetime import datetime
import copy

import optuna
from optuna import Trial
from optuna.samplers import TPESampler


from multiprocessing import Manager
import multiprocessing
from joblib import parallel_backend
#parser = argparse.ArgumentParser()

#parser.add_argument('--dset_id', required=True, type=int)
#parser.add_argument('--vision_dset', action = 'store_true')
#parser.add_argument('--task', required=True, type=str,choices = ['binary','multiclass','regression'])
#parser.add_argument('--cont_embeddings', default='MLP', type=str,choices = ['MLP','Noemb','pos_singleMLP'])
#parser.add_argument('--embedding_size', default=32, type=int)
#parser.add_argument('--transformer_depth', default=6, type=int)
#parser.add_argument('--attention_heads', default=8, type=int)
#parser.add_argument('--attention_dropout', default=0.1, type=float)
#parser.add_argument('--ff_dropout', default=0.1, type=float)
#parser.add_argument('--attentiontype', default='colrow', type=str,choices = ['col','colrow','row','justmlp','attn','attnmlp'])
#
#parser.add_argument('--optimizer', default='AdamW', type=str,choices = ['AdamW','Adam','SGD'])
#parser.add_argument('--scheduler', default='cosine', type=str,choices = ['cosine','linear'])
#
#parser.add_argument('--lr', default=0.0001, type=float)
#parser.add_argument('--epochs', default=100, type=int)
#parser.add_argument('--batchsize', default=256, type=int)
#parser.add_argument('--savemodelroot', default='./bestmodels', type=str)
#parser.add_argument('--run_name', default='testrun', type=str)
#parser.add_argument('--set_seed', default= 1 , type=int)
#parser.add_argument('--dset_seed', default= 1 , type=int)
#parser.add_argument('--active_log', action = 'store_true')
#
#parser.add_argument('--pretrain', action = 'store_true')
#parser.add_argument('--pretrain_epochs', default=50, type=int)
#parser.add_argument('--pt_tasks', default=['contrastive','denoising'], type=str,nargs='*',choices = ['contrastive','contrastive_sim','denoising'])
#parser.add_argument('--pt_aug', default=[], type=str,nargs='*',choices = ['mixup','cutmix'])
#parser.add_argument('--pt_aug_lam', default=0.1, type=float)
#parser.add_argument('--mixup_lam', default=0.3, type=float)
#
#parser.add_argument('--train_noise_type', default=None , type=str,choices = ['missing','cutmix'])
#parser.add_argument('--train_noise_level', default=0, type=float)
#
#parser.add_argument('--ssl_samples', default= None, type=int)
#parser.add_argument('--pt_projhead_style', default='diff', type=str,choices = ['diff','same','nohead'])
#parser.add_argument('--nce_temp', default=0.7, type=float)
#
#parser.add_argument('--lam0', default=0.5, type=float)
#parser.add_argument('--lam1', default=10, type=float)
#parser.add_argument('--lam2', default=1, type=float)
#parser.add_argument('--lam3', default=10, type=float)
#parser.add_argument('--final_mlp_style', default='sep', type=str,choices = ['common','sep'])
#
#parser.add_argument('--pretrain_save_path', default=None, type=str)

class model_opt(object):
    def __repr__(self):
        """Provide a detailed string representation of the object."""
        attributes = ', '.join([f"{key}={value!r}" for key, value in self.__dict__.items()])
        return f"{self.__class__.__name__}({attributes})"

    def __str__(self):
        """Provide a user-friendly string representation of the object."""
        attributes = ', '.join([f"{key}={value}" for key, value in self.__dict__.items()])
        return f"{self.__class__.__name__}: {attributes}"
def model_train_main(opt):


    #opt.device = 'cuda'
    #opt.dset_id = 42178
    #opt.vision_dset = True
    #opt.task = 'binary'
    #opt.cont_embeddings = 'MLP'
    #opt.embedding_size = 32
    #opt.transformer_depth = 6 
    #opt.attention_heads = 16
    #opt.attention_dropout = 0.4
    #opt.ff_dropout = 0.4
    #opt.attentiontype = 'col'
    #                    
    #opt.optimizer = 'AdamW'
    #opt.scheduler = 'cosine'
    #                    
    #opt.lr = 0.000001
    #opt.epochs = 100
    #opt.batchsize = 64
    #opt.savemodelroot = './bestmodels'
    #opt.run_name = 'testrun'
    #opt.set_seed = 1 
    #opt.dset_seed = 1
    ##opt.active_log = 
    #                    
    #opt.pretrain = True
    #opt.pretrain_epochs = 50
    #opt.pretrain_lr = 0.0001
    #opt.pt_tasks = ['constrastive','denoising']
    #opt.pt_aug = ['mixup', 'cutmix']
    #opt.pt_aug_lam = 0.1
    #opt.mixup_lam = 0.3
    #                    
    #opt.train_noise_type = ['missing','cutmix']
    #opt.train_noise_level = 0
    #                    
    #opt.ssl_samples = 32
    #opt.pt_projhead_style = 'diff'
    #opt.nce_temp = 0.7
    #                    
    #opt.lam0 = 0.5
    #opt.lam1 = 10
    #opt.lam2 = 1
    #opt.lam3 = 10 
    #opt.final_mlp_style = 'sep'
    #                    
    #opt.pretrain_save_path = './pretrain_model/test.dict.pt'



    print(opt)



    modelsave_path = os.path.join(os.getcwd(),opt.savemodelroot,opt.task,str(opt.dset_id),opt.run_name)
    opt.dtask = 'clf'
    #if opt.task == 'regression':
    #    opt.dtask = 'reg'
    #else:
    #    opt.dtask = 'clf'


    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}.")

    torch.manual_seed(opt.set_seed)
    os.makedirs(modelsave_path, exist_ok=True)

    print('Downloading and processing the dataset, it might take some time.')
    cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std \
        = data_prep_openml(opt.dset_id, opt.dset_seed,opt.task, datasplit=[.65, .15, .2])
    continuous_mean_std = np.array([train_mean,train_std]).astype(np.float32) 

    ##### Setting some hyperparams based on inputs and dataset
    _,nfeat = X_train['data'].shape




    train_ds = DataSetCatCon(X_train, y_train, cat_idxs,opt.dtask,continuous_mean_std)
    trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True,num_workers=4)

    valid_ds = DataSetCatCon(X_valid, y_valid, cat_idxs,opt.dtask, continuous_mean_std)
    validloader = DataLoader(valid_ds, batch_size=opt.batchsize, shuffle=False,num_workers=4)

    test_ds = DataSetCatCon(X_test, y_test, cat_idxs,opt.dtask, continuous_mean_std)
    testloader = DataLoader(test_ds, batch_size=opt.batchsize, shuffle=False,num_workers=4)

    y_dim = len(np.unique(y_train['data'][:,0]))
    cat_dims = np.append(np.array([1]),np.array(cat_dims)).astype(int) #Appending 1 for CLS token, this is later used to generate embeddings.



    model = SAINT(
    categories = tuple(cat_dims), 
    num_continuous = len(con_idxs),                
    dim = opt.embedding_size,                           
    dim_out = 1,                       
    depth = opt.transformer_depth,                       
    heads = opt.attention_heads,                         
    attn_dropout = opt.attention_dropout,             
    ff_dropout = opt.ff_dropout,                  
    mlp_hidden_mults = (4, 2),       
    cont_embeddings = opt.cont_embeddings,
    attentiontype = opt.attentiontype,
    final_mlp_style = opt.final_mlp_style,
    y_dim = y_dim
    )
    vision_dset = opt.vision_dset

    if y_dim == 2 and opt.task == 'binary':
        # opt.task = 'binary'
        criterion = nn.CrossEntropyLoss().to(device)
    elif y_dim > 2 and  opt.task == 'multiclass':
        # opt.task = 'multiclass'
        criterion = nn.CrossEntropyLoss().to(device)
    elif opt.task == 'regression':
        criterion = nn.MSELoss().to(device)
    else:
        raise'case not written yet'

    model.to(device)

    if opt.pretrain:
        print(f"[{opt.run_name} PRETRAINING START]")
        from pretraining import SAINT_pretrain
        model = SAINT_pretrain(model, cat_idxs,X_train,y_train, continuous_mean_std, opt,device)
        print(f"[{opt.run_name} PRETRAINING END]")
        if opt.pretrain_save_path:
            torch.save(model.state_dict(), opt.pretrain_save_path)

    if opt.ssl_samples is not None and opt.ssl_samples > 0 :
        print('We are in semi-supervised learning case')
        train_pts_touse = np.random.choice(X_train['data'].shape[0], opt.ssl_samples)
        X_train['data'] = X_train['data'][train_pts_touse,:]
        y_train['data'] = y_train['data'][train_pts_touse]
        
        X_train['mask'] = X_train['mask'][train_pts_touse,:]
        train_bsize = min(opt.ssl_samples//4,opt.batchsize)

        train_ds = DataSetCatCon(X_train, y_train, cat_idxs,opt.dtask,continuous_mean_std)
        trainloader = DataLoader(train_ds, batch_size=train_bsize, shuffle=True,num_workers=4)

    ## Choosing the optimizer

    if opt.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=opt.lr,
                              momentum=0.9, weight_decay=5e-4)
        from utils import get_scheduler
        scheduler = get_scheduler(opt, optimizer)
    elif opt.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(),lr=opt.lr)
    elif opt.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(),lr=opt.lr)
        
    scheduler = CosineAnnealingWarmUpRestarts(
        optimizer,
        T_0=100,
        T_mult=1,
        eta_max=opt.lr,
        T_up=10,
        gamma=0.5
    )
    early_stop = EarlyStopping(
        patience=20,
        delta=0.001,
        save_model=False,
    )
    best_valid_auroc = 0
    best_valid_accuracy = 0
    best_test_auroc = 0
    best_test_accuracy = 0
    best_valid_rmse = 100000
    print(f'[{opt.run_name}]]Training begins now.')
    for epoch in range(opt.epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            # x_categ is the the categorical data, with y appended as last feature. x_cont has continuous data. cat_mask is an array of ones same shape as x_categ except for last column(corresponding to y's) set to 0s. con_mask is an array of ones same shape as x_cont. 
            x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)
            if opt.train_noise_type is not None and opt.train_noise_level>0:
                noise_dict = {
                    'noise_type' : opt.train_noise_type,
                    'lambda' : opt.train_noise_level
                }
                if opt.train_noise_type == 'cutmix':
                    x_categ, x_cont = add_noise(x_categ,x_cont, noise_params = noise_dict)
                elif opt.train_noise_type == 'missing':
                    cat_mask, con_mask = add_noise(cat_mask, con_mask, noise_params = noise_dict)
            # We are converting the data to embeddings in the next step
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)           
            reps = model.transformer(x_categ_enc, x_cont_enc)
            # select only the representations corresponding to y and apply mlp on it in the next step to get the predictions.
            y_reps = reps[:,0,:]
            
            y_outs = model.mlpfory(y_reps)
            if opt.task == 'regression':
                loss = criterion(y_outs,y_gts) 
            else:
                loss = criterion(y_outs,y_gts.squeeze()) 
            loss.backward()
            optimizer.step()
            if opt.optimizer == 'SGD':
                scheduler.step()
            running_loss += loss.item()
        # val loss
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for _idx, data in enumerate(validloader,0):
                x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)
                _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)           
                reps = model.transformer(x_categ_enc, x_cont_enc)
                y_reps = reps[:,0,:]
                y_outs = model.mlpfory(y_reps)
                if opt.task == 'regression':
                    _v_loss = criterion(y_outs,y_gts)
                else:
                    _v_loss = criterion(y_outs,y_gts.squeeze())
                val_loss += _v_loss
            early_stop(val_loss, model)
            #import ipdb; ipdb.set_trace()
            print(f"[{opt.run_name}]epoch : {epoch}, train_loss : {round(running_loss,6)}, val_loss : {round(val_loss.item(),6)}")
            
            # validation case
            if opt.task in ['binary','multiclass']:
                val_accuracy, val_auroc = classification_scores(model, validloader, device, opt.task,vision_dset)

                print('[EPOCH %d] VALID ACCURACY: %.3f, VALID AUROC: %.3f' %
                    (epoch + 1, val_accuracy,val_auroc ))

            else:
                valid_rmse = mean_sq_error(model, validloader, device,vision_dset)    
                print('[EPOCH %d] VALID RMSE: %.3f' %
                    (epoch + 1, valid_rmse ))
        # print(running_loss)
        if epoch%5==0 or early_stop.early_stop == True:
                model.eval()
                with torch.no_grad():
                    if opt.task in ['binary','multiclass']:
                        accuracy, auroc = classification_scores(model, validloader, device, opt.task,vision_dset)
                        test_accuracy, test_auroc = classification_scores(model, testloader, device, opt.task,vision_dset)

                        print('[EPOCH %d] VALID ACCURACY: %.3f, VALID AUROC: %.3f' %
                            (epoch + 1, accuracy,auroc ))
                        print('[EPOCH %d] TEST ACCURACY: %.3f, TEST AUROC: %.3f' %
                            (epoch + 1, test_accuracy,test_auroc ))
                        if opt.task =='multiclass':
                            if accuracy > best_valid_accuracy:
                                best_valid_accuracy = accuracy
                                best_test_auroc = test_auroc
                                best_test_accuracy = test_accuracy
                                torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))
                        else:
                            if auroc > best_valid_auroc:
                                best_valid_auroc = auroc
                                best_test_auroc = test_auroc
                                best_test_accuracy = test_accuracy               
                                torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))

                    else:
                        valid_rmse = mean_sq_error(model, validloader, device,vision_dset)    
                        test_rmse = mean_sq_error(model, testloader, device,vision_dset)  
                        print('[EPOCH %d] VALID RMSE: %.3f' %
                            (epoch + 1, valid_rmse ))
                        print('[EPOCH %d] TEST RMSE: %.3f' %
                            (epoch + 1, test_rmse ))
                        if valid_rmse < best_valid_rmse:
                            best_valid_rmse = valid_rmse
                            best_test_rmse = test_rmse
                            torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))
                model.train()

        if early_stop.early_stop == True:
            print(f"[{opt.run_name}]Early Stopping; epoch : {epoch}, train_loss : {round(running_loss,6)}, val_loss : {round(val_loss.item(),6)}")
            break
        scheduler.step(epoch)
                    


    total_parameters = count_parameters(model)
    print('TOTAL NUMBER OF PARAMS: %d' %(total_parameters))
    if opt.task =='binary':
        print('AUROC on best model:  %.3f' %(best_test_auroc))
    elif opt.task =='multiclass':
        print('Accuracy on best model:  %.3f' %(best_test_accuracy))
    else:
        print('RMSE on best model:  %.3f' %(best_test_rmse))
    
    result_dict = {
        'train_loss' : float(running_loss),
        'val_loss'   : float(val_loss.item()),
        'val_acc'    : float(val_accuracy),
        'val_auroc'  : float(val_auroc),
        'test_acc'   : float(test_accuracy),
        'test_auroc' : float(test_auroc),
    }
    print(f"[{opt.run_name} FINSHED]")
    return result_dict



class Objective:
    def __init__(self, gpu_queue):
        self.gpu_queue = gpu_queue

    def __call__(self, trial):
        gpu_id = self.gpu_queue.get()
        time.sleep(random.randint(3,8)*0.1)

        trial.set_user_attr("GPU", int(gpu_id))


        opt = model_opt()
        opt.device = f'cuda:{gpu_id}'
        opt.dset_id = 42178
        opt.vision_dset = True
        opt.task = 'binary'
        opt.cont_embeddings = 'MLP'
        opt.embedding_size = trial.suggest_int("embedding_size", 16, 64)
        opt.transformer_depth = trial.suggest_int("transformer_depth", 4,16)
        opt.attention_heads = trial.suggest_int("attention_heads", 4,32)
        opt.attention_dropout = 0.4
        opt.ff_dropout = 0.4
        opt.attentiontype = 'col'
                            
        opt.optimizer = 'AdamW'
        opt.scheduler = 'cosine'
                            
        opt.lr = trial.suggest_float("lr", 1e-7, 1e-3)
        opt.epochs = trial.suggest_int("train_epochs",10, 1000)
        #opt.epochs = trial.suggest_int("train_epochs",1, 5)
        opt.batchsize = 64
        opt.savemodelroot = './bestmodels'
        opt.run_name = f'testrun-GPU:{gpu_id}'
        opt.set_seed = 1 
        opt.dset_seed = 1
        #opt.active_log = 
                            
        opt.pretrain = True
        opt.pretrain_epochs = trial.suggest_int("pretrain_epochs", 10, 500)
        #opt.pretrain_epochs = trial.suggest_int("pretrain_epochs", 1, 5)
        opt.pretrain_lr = trial.suggest_float("pretrian_lr", 1e-6, 1e-3)
        opt.pt_tasks = ['constrastive','denoising']
        opt.pt_aug = ['mixup', 'cutmix']
        opt.pt_aug_lam = 0.1
        opt.mixup_lam = 0.3
                            
        opt.train_noise_type = ['missing','cutmix']
        opt.train_noise_level = 0
                            
        opt.ssl_samples = 32
        opt.pt_projhead_style = 'diff'
        opt.nce_temp = 0.7
                            
        opt.lam0 = 0.5
        opt.lam1 = 10
        opt.lam2 = 1
        opt.lam3 = 10 
        opt.final_mlp_style = 'sep'
                            
        opt.pretrain_save_path = None
        
        result_dict = model_train_main(opt)

        trial.set_user_attr("train_loss", result_dict.get('train_loss'))
        trial.set_user_attr("val_loss", result_dict.get('val_loss'))
        trial.set_user_attr("val_acc", result_dict.get('val_acc'))
        trial.set_user_attr("val_auroc", result_dict.get('val_auroc'))
        trial.set_user_attr("test_acc", result_dict.get('test_acc'))
        trial.set_user_attr("test_auroc", result_dict.get('test_auroc'))
        
        
        time.sleep(random.randint(1,3)*0.1)
        self.gpu_queue.put(gpu_id)
        time.sleep(random.randint(1,3)*0.1)

        return result_dict['val_loss']


def run_study(n_trials, n_worker, gpu_list = [0,1,2,3], study_name=None):
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d_%H-%M")
    if study_name is not None:
        study_name = study_name
    else:
        study_name = f'saint_mdi_opt-{formatted_time}'

    sampler = TPESampler(seed=1234)
    study = optuna.create_study(
        study_name = study_name,
        direction="minimize",
        sampler=sampler,
        #storage=f"postgresql://admin:admin@192.168.100.51:21432/saint",
        storage=f"sqlite:///saint.db",
        load_if_exists=True,
    )
    
    gpu_pop = infinite_pop(gpu_list)
    with Manager() as manager:
        gpu_queue = manager.Queue()
        for i in range (n_worker):
            _gpu_id = copy.deepcopy(next(gpu_pop))
            gpu_queue.put(_gpu_id)
            print(f"Running GPU : {_gpu_id}")

        with parallel_backend("multiprocessing", n_jobs=n_worker):
            study.optimize(Objective(gpu_queue), n_trials=n_trials, n_jobs=n_worker)

    time.sleep(0.5)
    #study.optimize(objective, n_trials=100)





if __name__=='__main__':
    multiprocessing.set_start_method('spawn')
    n_trials = 200
    n_worker = 8
    gpu_list = [0,1,2,3]
    run_study(
        n_trials=n_trials,
        n_worker=n_worker,
        gpu_list=gpu_list,
        study_name = 'saint_mdi_opt-2024-05-06_22-33'
    )