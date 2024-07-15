import os
import sys
import logging
import random
import numpy as np
import copy
import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import shutil
from tqdm import tqdm

from options import args_parser
from dataloaders import dataset
from unet_base import UNET
from local_supervised import SupervisedLocalUpdate
from FedAvg import FedAvg, FedAvg_W, model_dist
from val_2D1 import test_single_volume3, test_single_volume1

from local_unsupervised import UnsupervisedLocalUpdate
from local_semisupervised import SemiSupervisedLocalUpdate


def split(dataset_L, dataset_U, id_1, id_2, id_3):
    dict_users_L, all_idxs_L = {}, [i for i in range(len(dataset_L))]
    dict_users_U, all_idxs_U = {}, [i for i in range(len(dataset_U))]

    for i in id_1:
        num_items_L = int(len(dataset_L)/4)
        dict_users_L[i] = set(np.random.choice(all_idxs_L, num_items_L, replace=False))
        all_idxs_L = list(set(all_idxs_L) - dict_users_L[i])

    for i in id_2:
        num_items_L = int(len(dataset_L)/5) - int((i-2)*45)########################################
        dict_users_L[i] = set(np.random.choice(all_idxs_L, num_items_L, replace=False))
        all_idxs_L = list(set(all_idxs_L) - dict_users_L[i])

        num_items_U = int(len(dataset_L)/4) - num_items_L
        dict_users_U[i-2] = set(np.random.choice(all_idxs_U, num_items_U, replace=False))
        all_idxs_U = list(set(all_idxs_U) - dict_users_U[i-2])

    for i in id_3:
        num_items_U = int(len(dataset_L)/4)
        dict_users_U[i-2] = set(np.random.choice(all_idxs_U, num_items_U, replace=False))
        all_idxs_U = list(set(all_idxs_U) - dict_users_U[i-2])

    return dict_users_L, dict_users_U


def create_model(args, ema=False):
    # Network definition
    model = UNET(args)
    # model = net_factory(net_type=args.model, in_chns=1,
    #                     class_num=num_classes)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def test(args, net_glob, val_dataset):
    model = net_glob

    if args.dataset_seg == 'vein':
        normalize = transforms.Normalize([0.475], [0.236])
    else:
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])

    test_dataset = val_dataset

    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, pin_memory=True)

    model.eval()

    metric_list = 0.0

    for i_batch, sampled_batch in enumerate(test_dataloader):
        metric_i = test_single_volume1(
            sampled_batch["image"], sampled_batch["label"], model, classes=args.num_classes)
        metric_list += np.array(metric_i)

    metric_list = metric_list / len(test_dataset)
    performance_dice = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    
    return performance_dice, mean_hd95


if __name__ == '__main__':
    args = args_parser()
    
    snapshot_path = "../model/{}_{}_labeled/{}".format(
    args.exp, args.labeled_num, args.model)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')

    shutil.copytree('.', snapshot_path + '/code',
                shutil.ignore_patterns(['.git', '__pycache__']))

    supervised_user_id = [0,1]
    par_supervised_user_id = [2,3,4,5]
    unsupervised_user_id = [6,7,8,9,10,11]

    print('done')
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # the below Normalize is from the sample of ImageNet dataset
    # normalize = transforms.Normalize([0.485, 0.456, 0.406],
    #                                  [0.229, 0.224, 0.225])

    # the below Normalize is for VESSEL-NIR dataset
    if args.dataset_seg == 'vein':
        normalize = transforms.Normalize([0.475], [0.236])# mean: 121.099/255, std: 60.305/255
    else:
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])

    all_dataset = dataset.BaseDataSets(base_dir=args.root_path,
                                       transform=transforms.Compose([transforms.ToTensor(), normalize]))

    train_dataset, val_dataset = torch.utils.data.random_split(all_dataset, [900, 100])

    #unlabeled_dataset = dataset.BaseDataSets(base_dir=args.root_path_unlabeled,
    #                                   transform=dataset.TransformTwice(
    #                                       transforms.Compose([
    #                                           transforms.ColorJitter(brightness=0.2), transforms.ToTensor(), normalize])))
                                               
    unlabeled_dataset = dataset.BaseDataSets(base_dir=args.root_path_unlabeled,
                                       transform=dataset.TransformTwice(
                                           transforms.Compose([transforms.ToTensor(), normalize])))

    dict_users_L, dict_users_U = split(train_dataset, unlabeled_dataset, supervised_user_id, par_supervised_user_id, unsupervised_user_id)

    net_glob = create_model(args)

    if len(args.gpu.split(',')) > 1:
        net_glob = torch.nn.DataParallel(net_glob).cuda()

    net_glob.train()
    w_glob = net_glob.state_dict()

    w_locals = []
    ful_w_locals = []
    par_w_locals = []
    un_w_locals = []

    trainer_locals = []
    net_locals = []
    optim_locals = []

    par_lab_trainer_locals = []
    par_sup_net_locals = []
    par_sup_optim_locals = []

    un_lab_trainer_locals = []
    un_sup_net_locals = []
    un_sup_optim_locals = []

    for i in supervised_user_id:
        trainer_locals.append(SupervisedLocalUpdate(args, train_dataset, dict_users_L[i], snapshot_path, i))
        w_locals.append(copy.deepcopy(w_glob))
        ful_w_locals.append(copy.deepcopy(w_glob))
        net_locals.append(copy.deepcopy(net_glob).cuda())
        optimizer = torch.optim.Adam(net_locals[i].parameters(), lr=args.base_lr)
        optim_locals.append(copy.deepcopy(optimizer.state_dict()))

    for i in par_supervised_user_id:
        par_lab_trainer_locals.append(SemiSupervisedLocalUpdate(args, train_dataset, dict_users_L[i], 
                                                                unlabeled_dataset, dict_users_U[i-2], snapshot_path, i))
        w_locals.append(copy.deepcopy(w_glob))
        par_w_locals.append(copy.deepcopy(w_glob))
        par_sup_net_locals.append(copy.deepcopy(net_glob).cuda())
        optimizer = torch.optim.Adam(par_sup_net_locals[i-2].parameters(), lr=args.base_lr)
        par_sup_optim_locals.append(copy.deepcopy(optimizer.state_dict()))

    for i in unsupervised_user_id:
        un_lab_trainer_locals.append(UnsupervisedLocalUpdate(args, unlabeled_dataset, dict_users_U[i-2], snapshot_path, i))
        w_locals.append(copy.deepcopy(w_glob))
        un_w_locals.append(copy.deepcopy(w_glob))
        un_sup_net_locals.append(copy.deepcopy(net_glob).cuda())
        optimizer = torch.optim.Adam(un_sup_net_locals[i-6].parameters(), lr=args.base_lr)
        un_sup_optim_locals.append(copy.deepcopy(optimizer.state_dict()))
        
    best_performance = 0.0

    for com_round in range(args.rounds):
        print("************* Comm round %d begins *************" % com_round)
        loss_locals = []

        for idx in supervised_user_id:
            local = trainer_locals[idx]
            optimizer = optim_locals[idx]

            w, loss, op = local.train(args, net_locals[idx], optimizer, val_dataset)
            w_locals[idx] = copy.deepcopy(w)
            ful_w_locals[idx] = copy.deepcopy(w)
            optim_locals[idx] = copy.deepcopy(op)
            loss_locals.append(loss)


        if com_round >= 0:
            for idx in par_supervised_user_id:
                local = par_lab_trainer_locals[idx - 2]
                optimizer = par_sup_optim_locals[idx - 2]

                if com_round == 0:
                    net_for_optim = FedAvg(ful_w_locals)
                elif com_round > 0:
                    net_for_optim = net_glob.state_dict()

                w, loss, op = local.train(args, par_sup_net_locals[idx-2], optimizer, val_dataset, com_round, net_for_optim)
                w_locals[idx] = copy.deepcopy(w)
                par_w_locals[idx-2] = copy.deepcopy(w)
                par_sup_optim_locals[idx-2] = copy.deepcopy(op)
                loss_locals.append(loss)
            
            
            for idx in unsupervised_user_id:
                local = un_lab_trainer_locals[idx - 6]
                optimizer = un_sup_optim_locals[idx - 6]

                if com_round == 0:
                    net_for_optim = FedAvg(ful_w_locals)
                elif com_round > 0:
                    net_for_optim = net_glob.state_dict()

                w, loss, op = local.train(args, un_sup_net_locals[idx-6], optimizer, val_dataset, com_round, net_for_optim)
                w_locals[idx] = copy.deepcopy(w)
                un_w_locals[idx-6] = copy.deepcopy(w)
                un_sup_optim_locals[idx-6] = copy.deepcopy(op)
                loss_locals.append(loss)

            #weights
            w_avg_all = FedAvg(w_locals)
            w_avg_ful = FedAvg(ful_w_locals)
            w_avg_par = FedAvg(par_w_locals)
            w_avg_un = FedAvg(un_w_locals)

            dist_scale_f = args.dist_scale

            #all
            dist_list = []
            for cli_idx in range(12):
                dist = model_dist(w_locals[cli_idx], w_avg_all)
                dist_list.append(dist)
            clt_freq_this_meta = [np.exp(-dist_list[i] * dist_scale_f) for i in range(12)]

            #fully_supervised
            dist_list_ful = []
            for cli_idx in range(2):
                dist = model_dist(ful_w_locals[cli_idx], w_avg_ful)
                dist_list_ful.append(dist)
            clt_freq_this_ful_meta = [np.exp(-dist_list_ful[i] * dist_scale_f) for i in range(2)]

            #partially_supervised
            dist_list_par = []
            for cli_idx in range(4):
                dist = model_dist(par_w_locals[cli_idx], w_avg_par)
                dist_list_par.append(dist)
            clt_freq_this_par_meta = [np.exp(-dist_list_par[i] * dist_scale_f) for i in range(4)]

            #un_supervised
            dist_list_un = []
            for cli_idx in range(6):
                dist = model_dist(un_w_locals[cli_idx], w_avg_un)
                dist_list_un.append(dist)
            clt_freq_this_un_meta = [np.exp(-dist_list_un[i] * dist_scale_f) for i in range(6)]

            ###############################################################################
            logging.info('clt_freq_this_meta : {}'.format(clt_freq_this_meta))
            logging.info('dist_list : {}'.format(dist_list))
            logging.info('clt_freq_this_ful_meta : {}'.format(clt_freq_this_ful_meta))
            logging.info('dist_list_ful : {}'.format(dist_list_ful))
            logging.info('clt_freq_this_par_meta : {}'.format(clt_freq_this_par_meta))
            logging.info('dist_list_par : {}'.format(dist_list_par))
            logging.info('clt_freq_this_un_meta : {}'.format(clt_freq_this_un_meta))
            logging.info('dist_list_un : {}'.format(dist_list_un))
            ###############################################################################
            
            #weight calculation
            list_of_user_id = [supervised_user_id, par_supervised_user_id, unsupervised_user_id]
            list_of_clt_freq = [clt_freq_this_ful_meta, clt_freq_this_par_meta, clt_freq_this_un_meta]
            list_weight = [ful_w_locals, par_w_locals, un_w_locals]

            total_meta_length = 0

            w_per_meta = []

            for id in range(len(list_of_user_id)):

                total = sum([clt_freq_this_meta[i] for i in list_of_user_id[id]]) + sum(list_of_clt_freq[id])
                clt_freq_this_meta_dist = []    
                
                ###############################################################################
                logging.info('id : {}'.format(id))
                logging.info('total : {}'.format(total))
                ###############################################################################
            
                for idx in range(len(list_of_user_id[id])):
                    weight_client = (clt_freq_this_meta[idx+total_meta_length] + list_of_clt_freq[id][idx]) / total
                    ###############################################################################
                    logging.info('idx : {}'.format(idx))
                    logging.info('weight_client : {}'.format(weight_client))
                    ###############################################################################
                    clt_freq_this_meta_dist.append(weight_client)

                total_meta_length += len(list_of_user_id[id])

                clt_freq_this_meta_round = clt_freq_this_meta_dist

                assert sum(clt_freq_this_meta_round) - 1.0 <= 1e-3, "Error: sum(freq) != 0"
                w_this_meta = FedAvg_W(list_weight[id], clt_freq_this_meta_round)

                w_per_meta.append(w_this_meta)

            with torch.no_grad():
                w_glob = FedAvg(w_per_meta)

            net_glob.load_state_dict(w_glob)

            list_net = [net_locals, par_sup_net_locals, un_sup_net_locals]

            for id in range(len(list_of_user_id)):
                net_temp = list_net[id]
                for i in range(len(list_of_user_id[id])):
                    net_temp[i].load_state_dict(w_glob)

            loss_avg = sum(loss_locals) / len(loss_locals)
            logging.info('Round: %d, Loss Avg: %f' % (com_round, loss_avg))

        else:
            with torch.no_grad():
                w_glob = FedAvg(ful_w_locals)

            net_glob.load_state_dict(w_glob)
            
            list_of_user_id = [supervised_user_id, par_supervised_user_id, unsupervised_user_id]
            
            list_net = [net_locals, par_sup_net_locals, un_sup_net_locals]

            for id in range(len(list_of_user_id)):
                net_temp = list_net[id]
                for i in range(len(list_of_user_id[id])):
                    net_temp[i].load_state_dict(w_glob)
            
        #evaluation
        net_glob.eval()

        performance_dice, mean_hd95 = test(args, net_glob, val_dataset)

        logging.info('epoch: %d, mean_dice: %f, mean_hd95: %f' 
                     % (com_round, performance_dice, mean_hd95))

        if performance_dice > best_performance:
            best_performance = performance_dice

            save_mode_path = os.path.join(snapshot_path, 'epoch_' + 'dice' + '.pth')
            
            #save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(com_round) + 'dice_' + str(round(best_performance, 4)) + '.pth')
                
            torch.save({'state_dict': net_glob.module.state_dict()}, save_mode_path)

        net_glob.train()