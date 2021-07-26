import torch
from util.torch_dist_sum import *
from data.imagenet import *
from data.augmentation import *
import torch.nn as nn
from util.meter import *
from network.wcl import WCL
import time
from util.accuracy import accuracy
from math import sqrt
import math
from util.LARS import LARS
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size-pergpu', type=int, default=128)
parser.add_argument('--epochs', type=int, default=100)
args = parser.parse_args()
print(args)

epochs = args.epochs
warm_up = 10

def adjust_learning_rate(optimizer, epoch, base_lr, i, iteration_per_epoch):
    T = epoch * iteration_per_epoch + i
    warmup_iters = warm_up * iteration_per_epoch
    total_iters = (epochs - warm_up) * iteration_per_epoch

    if epoch < warm_up:
        lr = base_lr * 1.0 * T / warmup_iters
    else:
        T = T - warmup_iters
        lr = 0.5 * base_lr * (1 + math.cos(1.0 * T / total_iters * math.pi))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(train_loader, model, local_rank, rank, criterion, optimizer, epoch, iteration_per_epoch, base_lr):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    graph_losses = AverageMeter('graph', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, graph_losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (img1, img2) in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, base_lr, i, iteration_per_epoch)
        data_time.update(time.time() - end)

        if local_rank is not None:
            img1 = img1.cuda(local_rank, non_blocking=True)
            img2 = img2.cuda(local_rank, non_blocking=True)

        # compute output
        output, target, graph_loss = model(img1, img2)
        ce_loss = criterion(output, target)
        loss = ce_loss + graph_loss

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(ce_loss.item(), img1.size(0))
        graph_losses.update(graph_loss.item(), img1.size(0))
        top1.update(acc1[0], img1.size(0))
        top5.update(acc5[0], img1.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0 and rank == 0:
            progress.display(i)


def main():
    from torch.nn.parallel import DistributedDataParallel
    from util.dist_init import dist_init
    
    rank, local_rank, world_size = dist_init()
    batch_size = args.batch_size_pergpu
    num_workers = 8
    base_lr = 0.075 * sqrt(batch_size * world_size)

    model = WCL()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda()
    
    param_dict = {}
    for k, v in model.named_parameters():
        param_dict[k] = v

    bn_params = [v for n, v in param_dict.items() if ('bn' in n or 'bias' in n)]
    rest_params = [v for n, v in param_dict.items() if not ('bn' in n or 'bias' in n)]

    optimizer = torch.optim.SGD([{'params': bn_params, 'weight_decay': 0, 'ignore': True },
                                {'params': rest_params, 'weight_decay': 1e-6, 'ignore': False}], 
                                lr=base_lr, momentum=0.9, weight_decay=1e-6)

    optimizer = LARS(optimizer, eps=0.0)
    model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    torch.backends.cudnn.benchmark = True

    weak_aug_train_dataset = ImagenetContrastive(aug=moco_aug, max_class=1000)
    weak_aug_train_sampler = torch.utils.data.distributed.DistributedSampler(weak_aug_train_dataset)
    weak_aug_train_loader = torch.utils.data.DataLoader(
        weak_aug_train_dataset, batch_size=batch_size, shuffle=(weak_aug_train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=weak_aug_train_sampler, drop_last=True)

    train_dataset = ImagenetContrastive(aug=simclr_aug, max_class=1000)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    iteration_per_epoch = train_loader.__len__()
    criterion = nn.CrossEntropyLoss().cuda(local_rank)

    checkpoint_path = 'checkpoints/wcl-{}.pth'.format(epochs)
    print('checkpoint_path:', checkpoint_path)
    if os.path.exists(checkpoint_path):
        checkpoint =  torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0
    

    model.train()
    for epoch in range(start_epoch, epochs):
        if epoch < warm_up:
            weak_aug_train_sampler.set_epoch(epoch)
            train(weak_aug_train_loader, model, local_rank, rank, criterion, optimizer, epoch, iteration_per_epoch, base_lr)
        else:
            train_sampler.set_epoch(epoch)
            train(train_loader, model, local_rank, rank, criterion, optimizer, epoch, iteration_per_epoch, base_lr)
        
        if rank == 0:
            torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1
            }, checkpoint_path)

if __name__ == "__main__":
    main()
