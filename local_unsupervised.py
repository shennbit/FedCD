import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import torch.optim
from options import args_parser
import copy
from utils import losses
from torch.nn.modules.loss import CrossEntropyLoss
from tensorboardX import SummaryWriter
import logging
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from val_2D1 import test_single_volume3, test_single_volume1
from unet_base import UNET
from ramp import LinearRampUp
from utils_SimPLE import label_guessing, sharpen

args = args_parser()


def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


# alpha=0.999
def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


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


def create_model(args, ema=False):
    # Network definition
    model = UNET(args)
    # model = net_factory(net_type=args.model, in_chns=1,
    #                     class_num=num_classes)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
         
        return len(self.idxs)

    def __getitem__(self, item):
         
        sample = self.dataset[self.idxs[item]]
        return sample

class UnsupervisedLocalUpdate(object):
    def __init__(self, args, dataset, idxs, snapshot_path, user_id):
          
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size = args.batch_size, shuffle = True)
          
        self.snapshot_path = snapshot_path
        self.user_id = user_id
        self.base_lr = args.base_lr

        net = create_model(args, True)

        if len(args.gpu.split(',')) > 1:
            net = torch.nn.DataParallel(net).cuda()

        self.ema_model = net

        self.flag = True
        
        self.max_grad_norm = args.max_grad_norm

        self.max_warmup_step = round(len(idxs) / args.batch_size) * args.num_warmup_epochs
        self.ramp_up = LinearRampUp(length=self.max_warmup_step)

    def train(self, args, net, op_dict, val_dataset, epoch, net_for_optim):
        self.model = copy.deepcopy(net)
        self.model.train()
        self.ema_model.train()

        ################################################
        model_global = copy.deepcopy(net)
        model_global.load_state_dict(net_for_optim)
        model_global.eval()
        ################################################

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.base_lr)
        self.optimizer.load_state_dict(op_dict)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr

        self.epoch = epoch
        if self.flag:
            self.ema_model.load_state_dict(copy.deepcopy(net).state_dict())
            #self.ema_model.load_state_dict(copy.deepcopy(net))
            self.flag = False
            logging.info('EMA model initialized')

        writer = SummaryWriter(self.snapshot_path + '/log')
        logging.info("{} iterations per id".format(len(self.ldr_train)))
        
        iter_num = 0
        best_performance = 0.0
        max_epoch = args.max_iterations // len(self.ldr_train) + 1
        iterator = tqdm(range(max_epoch), ncols=70)
            
        for epoch_num in iterator:
            
            logging.info('\n')
            
            for i_batch_u, sampled_batch_u in enumerate(self.ldr_train):
                weak_aug_batch = [sampled_batch_u['image'][version].cuda() for version in range(len(sampled_batch_u['image']))]
                with torch.no_grad():
                    guessed = label_guessing(self.ema_model, [weak_aug_batch[0]])################
                    sharpened = sharpen(guessed)

                logits_str = self.model(weak_aug_batch[1])
                probs_str = torch.sigmoid(logits_str)

                loss_u = torch.mean(losses.softmax_mse_loss(probs_str, sharpened, sigmoid=True))
                #loss_u = torch.sum(losses.softmax_mse_loss(probs_str, sharpened)) / args.batch_size

                ramp_up_value = self.ramp_up(current=self.epoch)

                ################################################
                with torch.no_grad():
                    logits_global = model_global(weak_aug_batch[1])
                    probs_global = torch.sigmoid(logits_global)

                loss_g = torch.mean(losses.softmax_mse_loss(probs_str, probs_global, sigmoid=True))
                #loss_g = torch.sum(losses.softmax_mse_loss(probs_str, probs_global)) / args.batch_size
                ################################################

                loss = ramp_up_value * args.lambda_u * (loss_u + loss_g)

                #entropy_loss = losses.entropy_loss(probs_str, C=2)

                #consistency_loss = torch.mean((probs_str - ema_output_soft)**2)

                # loss = consistency_loss
                    
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                              max_norm=self.max_grad_norm)
                self.optimizer.step()

                update_ema_variables(self.model, self.ema_model, args.ema_decay, iter_num)

                lr_ = self.base_lr * (1.0 - iter_num / (max_epoch * len(self.ldr_train))) ** 0.9
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr_

                iter_num = iter_num + 1
                
                logging.info('user_id: %d, iteration: %d, loss: %f, loss_u: %f, loss_g: %f'
                             % (self.user_id, iter_num, loss.item(), loss_u, loss_g))

                if iter_num > 0 and iter_num % 50 == 0:
                    
                    self.model.eval()

                    performance_dice, mean_hd95 = test(args, self.model, val_dataset)

                    logging.info('VAL')
                    logging.info('user_id: %d, iteration: %d, mean_dice: %f, mean_hd95: %f' 
                                 % (self.user_id, iter_num, performance_dice, mean_hd95))

                    self.model.train()
                    
                    self.ema_model.eval()

                    performance_ema_dice, mean_ema_hd95 = test(args, self.ema_model, val_dataset)

                    logging.info('user_id: %d, iteration: %d, mean_ema_dice: %f, mean_ema_hd95: %f' 
                                 % (self.user_id, iter_num, performance_ema_dice, mean_ema_hd95))
                                 
                    self.ema_model.train()
                    
                    global net_local_best
                    
                    global loss_local_best
                    
                    global optimizer_local_best
                                 
                    if performance_dice > performance_ema_dice and performance_ema_dice > best_performance:

                        best_performance = performance_dice
                        
                        net_local_best = copy.deepcopy(self.model)
 
                        loss_local_best = loss

                        optimizer_local_best = copy.deepcopy(self.optimizer)
                        
                    elif performance_ema_dice > performance_dice and performance_dice > best_performance:
                    
                        best_performance = performance_ema_dice
                        
                        net_local_best = copy.deepcopy(self.ema_model)
                        
                        loss_local_best = loss

                        optimizer_local_best = copy.deepcopy(self.optimizer)
 
        writer.close()
        iterator.close()

        return net_local_best.state_dict(), loss_local_best, copy.deepcopy(optimizer_local_best.state_dict())