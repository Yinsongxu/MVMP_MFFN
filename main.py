import argparse
import torch
from torchvision import transforms
import torchvision
from model import Cgnet
from engine import Engine
from data import ImageDataManager
from optim import *
from utils import WarmupMultiStepLR
def config():
    parser = argparse.ArgumentParser(description='unsupervised data augmentation')
    parser.add_argument('--train-batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--learning-rate', type=float, default=0.003)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--logiinterval', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='log')
    parser.add_argument('--dataset-dir', type=str, default='dataset')
    parser.add_argument('--dataset', type=str, default='market1501')
    parser.add_argument('--out-channels', default=2048, type=int )
    parser.add_argument('--print-freq', default=20, type=int )
    parser.add_argument('--eval-freq', default=10, type=int )
    parser.add_argument('--num_instances', default=8, type=int )
    parser.add_argument('--k', default=10, type=int )
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=128)
    return parser.parse_args()

def main():
    '''transform = transforms.Compose([transforms.ToTensor()
     ,transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])'''
    args = config()
    args.cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    datamanager = ImageDataManager(root='',
                    sources='market1501',height=256,width=128, 
                    combineall=False, 
                    batch_size_train=args.train_batch_size,
                    batch_size_test=args.test_batch_size,
                    train_sampler='RandomIdentitySampler',
                    num_instances=args.num_instances,
                    transforms=['random_flip', 'random_crop', 'random_erase'])
    model = Cgnet(datamanager.num_train_pids, args.out_channels, k=args.k,alpha=args.alpha)
    print(args.k)
    if args.cuda:
       model = model.cuda()
    
    optimizer = build_optimizer(
        model,
        optim='adam',
        lr=0.00035
    )
    scheduler = WarmupMultiStepLR(optimizer,[40,70],0.1,0.01,10)
    enigne = Engine(model, optimizer, datamanager, scheduler,k=args.k, gamma=args.gamma)
    enigne.run(save_dir=args.save_dir, max_epoch= args.epochs, eval_freq=args.eval_freq, print_freq=args.print_freq,test_only=True)

main()
