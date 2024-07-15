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

from val_2D1 import test_single_volume3, test_single_volume1

args = args_parser()


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


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
         
        return len(self.idxs)

    def __getitem__(self, item):
         
        sample = self.dataset[self.idxs[item]]
        return sample

class SupervisedLocalUpdate(object):
    def __init__(self, args, dataset, idxs, snapshot_path, user_id):
          
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size = args.batch_size, shuffle = True)
          
        self.snapshot_path = snapshot_path
        self.user_id = user_id
        self.base_lr = args.base_lr
         
    def train(self, args, net, op_dict, val_dataset):
        net.train()
        self.optimizer = torch.optim.Adam(net.parameters(), lr=args.base_lr)
        self.optimizer.load_state_dict(op_dict)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr

        ce_loss = CrossEntropyLoss()
        dice_loss = losses.DiceLoss(args.num_classes)

        writer = SummaryWriter(self.snapshot_path + '/log')
        logging.info("{} iterations per id".format(len(self.ldr_train)))
        
        iter_num = 0
        best_performance = 0.0
        max_epoch = args.max_iterations // len(self.ldr_train) + 1
        iterator = tqdm(range(max_epoch), ncols=70)
            
        for epoch_num in iterator:
            
            logging.info('\n')
            
            for i_batch_l, sampled_batch_l in enumerate(self.ldr_train):
                volume_batch, label_batch = sampled_batch_l['image'], sampled_batch_l['label']
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

                outputs = net(volume_batch)
                outputs_soft = torch.sigmoid(outputs)
            
                loss_ce = ce_loss(outputs, label_batch[:].long())
                loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            
                supervised_loss = loss_dice + loss_ce
                    
                loss = supervised_loss
            
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                lr_ = self.base_lr * (1.0 - iter_num / (max_epoch * len(self.ldr_train))) ** 0.9
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr_

                iter_num = iter_num + 1
                
                logging.info(
                    'user_id: %d, iteration: %d, loss: %f, loss_ce: %f, loss_dice: %f' %
                    (self.user_id, iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

                if iter_num > 0 and iter_num % 50 == 0:

                    net.eval()

                    performance_dice, mean_hd95 = test(args, net, val_dataset)

                    logging.info('VAL')
                    logging.info('user_id: %d, iteration: %d, mean_dice: %f, mean_hd95: %f' 
                                 % (self.user_id, iter_num, performance_dice, mean_hd95))

                    if performance_dice > best_performance:

                        best_performance = performance_dice

                        global net_local_best
                        
                        net_local_best = copy.deepcopy(net)
                        
                        global loss_local_best

                        loss_local_best = loss
                        
                        global optimizer_local_best

                        optimizer_local_best = copy.deepcopy(self.optimizer)

                    net.train()
 
        writer.close()
        iterator.close()

        return net_local_best.state_dict(), loss_local_best, copy.deepcopy(optimizer_local_best.state_dict())