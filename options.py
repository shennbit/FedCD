import argparse

def args_parser():
     parser = argparse.ArgumentParser()
     ### new
     parser.add_argument('--exp', type=str, default='VEIN/UNET_FL', help='experiment_name')
     parser.add_argument('--labeled_num', type=int, default=1000, help='labeled data')
     parser.add_argument('--model', type=str, default='unet_fl', help='model_name')
     parser.add_argument('--num_classes', type=int, default=2, help='output channel of network')
     parser.add_argument('--gpu', type=str,  default='0,1', help='GPU to use')
     parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
     parser.add_argument('--seed', type=int,  default=1337, help='random seed')
     parser.add_argument('--root_path', type=str, 
                         default='/data3/vein/vein-train-1000-480-640', help='dataset root dir')
     parser.add_argument('--num_users', type=int,  default=10, help='local epoch')
     parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
     parser.add_argument('--base_lr', type=float,  default=1e-4, help='segmentation network learning rate')
     parser.add_argument('--max_iterations', type=int, default=1000, help='maximum epoch number to train')
     parser.add_argument('--rounds', type=int,  default=50, help='local epoch')
     parser.add_argument('--local_ep', type=int,  default=1, help='local epoch')
     parser.add_argument('--dataset_seg', type=str, default='vein', help='dataset')
     parser.add_argument('--in_chan', type=int, default=1, help='input channel')
     parser.add_argument('--base_chan', type=int, default=32, help='base channel')
     parser.add_argument('--out_chan', type=int, default=2, help='out channel')
     parser.add_argument('--reduce_size', type=list, default=[9,12], help='reduce size')
     #parser.add_argument('--local_train', type=int, default=1000, help='local train time')

     parser.add_argument('--root_path_unlabeled', type=str, 
                         default='/data3/vein/vein-train-unlabel-2000-480-640', help='dataset root dir of unlabeled data')
     parser.add_argument('--each_length', type=int, default=225, help='data number each length')
     parser.add_argument('--dist_scale', type=int, default=0.01, help='scale factor when computing model distance')
     parser.add_argument("--max_grad_norm",
                        dest="max_grad_norm",
                        type=float,
                        default=5,
                        help="max gradient norm allowed (used for gradient clipping)")
     parser.add_argument('--num-warmup-epochs',
                        '--num-warm-up-epochs',
                        dest="num_warmup_epochs",
                        default=0,
                        type=int,
                        help='number of warm-up epochs for unsupervised loss ramp-up during training'
                             'set to 0 to disable ramp-up')
     parser.add_argument('--lambda_u', type=float, default=100, help='start_epoch')
     parser.add_argument('--ema_decay', type=float, default=0.999, help='ema_decay')
     parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
     parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')

     parser.add_argument('--labeled_bs', type=int, default=4, help='labeled_batch_size per gpu')
     parser.add_argument('--ict_alpha', type=int, default=0.2, help='ict_alpha')

     args = parser.parse_args()
     return args