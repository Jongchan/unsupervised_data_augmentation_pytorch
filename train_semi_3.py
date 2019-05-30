import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

from wideresnet import WideResNet
import numpy as np



parser = argparse.ArgumentParser()
parser.add_argument('--dataset',            type=str,       default='cifar10')
parser.add_argument('--arch',               type=str,       default='WRN-28-2')
parser.add_argument('--name',               type=str,       required=True)

# HYPER PARAMS
parser.add_argument('--dropout-rate',       type=float,     default=0.3)
parser.add_argument('--lr',                 type=float,     default=3e-3)
parser.add_argument('--batch-size',         type=int,       default=100)
parser.add_argument('--l1-reg',             type=float,     default=1e-6)
parser.add_argument('--l2-reg',             type=float,     default=1e-4)
parser.add_argument('--lr-decay-rate',      type=float,     default=0.2)
parser.add_argument('--max-iter',           type=int,       default=500000)
parser.add_argument('--lr-decay-at',        type=float,     default=400000)
parser.add_argument('--normalization',      type=str,       default='ZCA')
parser.add_argument('--gaussian-noise-level',type=float,    default=0.15)
parser.add_argument('--num-workers',        type=int,       default=20)
parser.add_argument('--print-freq',         type=int,       default=100)
parser.add_argument('--split',              type=int,       default=0)
parser.add_argument('--eval-iter',          type=int,       default=10000)

parser.add_argument('--UDA',                action='store_true')
parser.add_argument('--UDA-default-preproc',action='store_true')
parser.add_argument('--TSA',                type=str,       default=None)
parser.add_argument('--batch-size-unsup',   type=int,       default=960)
parser.add_argument('--unsup-loss-weight',  type=float,     default=1.0)

def TSA_th(cur_step):
    global args
    num_classes = 10
    if args.TSA == 'linear':
        th = float(cur_step) / float(args.max_iter) * (1-1 / float(num_classes)) + 1 / float(num_classes)
    elif args.TSA == 'log':
        th = (1 - np.exp(- float(cur_step) / float(args.max_iter) * 5)) * (1 - 1 / float(num_classes)) + 1 / float(num_classes)
    elif args.TSA == 'exp':
        th = np.exp( (float(cur_step) / float(args.max_iter) - 1) * 5) * (1 - 1 / float(num_classes)) + 1 / float(num_classes)
    else:
        th = 1.0
    return th

class GaussianNoise(object):
    def __init__(self, noise_level):
        self.noise_level = noise_level
    def __call__(self, img):
        return img + img.clone().normal_(0, self.noise_level)

def global_contrast_normalize(X, scale=55., min_divisor=1e-8):
    X = X.view(X.size(0), -1)
    X = X - X.mean(dim=1, keepdim=True)

    normalizers = torch.sqrt( torch.pow( X, 2).sum(dim=1, keepdim=True)) / scale
    normalizers[normalizers < min_divisor] = 1.
    X /= normalizers

    return X

class ZCA(object):
    def __init__(self, zca_params):
        self.meanX = torch.FloatTensor(zca_params['meanX']).unsqueeze(0).cuda()
        self.W = torch.FloatTensor(zca_params['W']).cuda()

    def __call__(self, sample):
        return torch.matmul( sample - self.meanX, self.W ).view(sample.size(0), 3,32,32)

def main():
    global args, best_prec1, exp_dir

    best_prec1 = 0
    args = parser.parse_args()

    global zca
    zca_params = torch.load('./zca_params.pth')
    zca = ZCA(zca_params)
    global gaussian
    gaussian = GaussianNoise(args.gaussian_noise_level)

    exp_dir = os.path.join('experiments_3', args.name)
    if os.path.exists(exp_dir):
        print ("same experiment exist...")
        #return
    else:
        os.makedirs(exp_dir)

    # DATA SETTINGS
    global dataset_train, dataset_test
    if args.dataset == 'cifar10':
        if args.normalization == 'MEAN_STD':
            import cifar_semi as cifar
        elif args.normalization == 'ZCA_v1':
            import cifar_semi_zca as cifar
        elif args.normalization == 'ZCA_v2':
            import cifar_semi_zca_v2 as cifar
        elif args.normalization == 'ZCA_v3':
            import cifar_semi_zca_v3 as cifar
        else:
            raise NotImplementedError("normalization {} is not implemented.".format(args.normalization))

        dataset_train = cifar.CIFAR10(args, train=True)
        dataset_test = cifar.CIFAR10(args, train=False)
    if args.UDA:
        # loader for UDA
        dataset_train_uda = cifar.CIFAR10(args, True, True)
        uda_loader    = torch.utils.data.DataLoader( dataset_train_uda,   batch_size=args.batch_size_unsup, shuffle=True,   num_workers=args.num_workers, pin_memory=True )
        iter_uda = iter(uda_loader)
    else:
        iter_uda = None
        
    train_loader, test_loader = initialize_loader()

    # MODEL SETTINGS
    if args.arch == 'WRN-28-2':
        model = WideResNet(28, [100,10][int(args.dataset=='cifar10')], 2, dropRate=args.dropout_rate)
        model = torch.nn.DataParallel(model.cuda())
    else:
        raise NotImplementedError('arch {} is not implemented'.format(args.arch))

    optimizer = torch.optim.Adam( model.parameters(), lr=args.lr, weight_decay=args.l2_reg )

    global batch_time, losses_sup, losses_unsup, top1, losses_l1, losses_unsup
    batch_time, losses_sup, losses_unsup, top1, losses_l1, losses_unsup = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    t = time.time()
    model.train()
    iter_sup = iter(train_loader)
    for train_iter in range(args.max_iter):
        # TRAIN
        lr = adjust_learning_rate(optimizer, train_iter + 1)
        train(model, iter_sup, optimizer, train_iter, data_iterator_uda=iter_uda)

        # LOGGING
        if (train_iter+1) % args.print_freq == 0:
            print('ITER: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'L1 Loss {l1_loss.val:.4f} ({l1_loss.avg:.4f})\t'
                  'Unsup Loss {unsup_loss.val:.4f} ({unsup_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Learning rate {2} TSA th {3}'.format(
                      train_iter, args.max_iter, lr, TSA_th(train_iter),
                      batch_time=batch_time,
                      loss=losses_sup, 
                      l1_loss=losses_l1,
                      unsup_loss=losses_unsup,
                      top1=top1))

        if (train_iter+1)%args.eval_iter == 0 or train_iter+1 == args.max_iter:
            # EVAL
            print ("evaluation at iter {}".format(train_iter))
            prec1 = test(model, test_loader)

            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'iter': train_iter + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)
            print ("* Best accuracy: {}".format(best_prec1))
            eval_interval_time = time.time() - t; t = time.time()
            print ("total {} sec for {} iterations".format(eval_interval_time, args.eval_iter))
            seconds_remaining = eval_interval_time / float(args.eval_iter) * (args.max_iter - train_iter)
            print ("{}:{}:{} remaining".format( int(seconds_remaining // 3600), int( (seconds_remaining % 3600) // 60), int(seconds_remaining % 60)))
            model.train()
            batch_time, losses_sup, losses_unsup, top1, losses_l1, losses_unsup = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
            iter_sup = iter(train_loader)
            if iter_uda is not None:
                iter_uda = iter(uda_loader)

def train(model, data_iterator, optimizer, iteration, data_iterator_uda=None):
    global args
    global batch_time, losses_sup, top1, losses_l1, losses_unsup
    t = time.time()

    input, target = next(data_iterator)
    input, target = input.cuda(), target.cuda().long()

    global zca
    input = zca( global_contrast_normalize( input ) )
    global gaussian
    input = gaussian( input )

    output = model(input)

    tsa_th = TSA_th( iteration )

    prec1 = accuracy(output.data, target, topk=(1,))[0]


    
    # Loss calculation with TSA
    num_classes = 10 if args.dataset=='cifar10' else 100
    target_onehot = torch.FloatTensor( input.size(0), num_classes ).cuda()
    target_onehot.zero_()
    target_onehot.scatter_(1, target.unsqueeze(1), 1)
    output_softmax = torch.nn.functional.softmax( output, dim=1 ).detach()
    gt_softmax = (target_onehot * output_softmax).sum(dim=1)
    loss_mask = (gt_softmax <= tsa_th).float()
    loss_sup = torch.sum( torch.nn.functional.cross_entropy(output, target, reduction='none') * loss_mask ) / (loss_mask.sum()+1e-6)
    #loss_sup = torch.nn.functional.cross_entropy(output, target)
    if args.UDA:
        input_unsup, input_unsup_aug = next(data_iterator_uda)
        input_unsup = input_unsup.cuda()
        input_unsup_aug = input_unsup_aug.cuda()
        input_unsup = gaussian( zca( global_contrast_normalize( input_unsup ) ) )
        input_unsup_aug = gaussian( zca( global_contrast_normalize( input_unsup_aug) ) )
        output_unsup = model(input_unsup)
        output_unsup_aug = model(input_unsup_aug)

        loss_unsup = torch.nn.functional.kl_div( 
                                torch.nn.functional.log_softmax(output_unsup_aug, dim=1), 
                                torch.nn.functional.softmax(output_unsup, dim=1).detach(),
                                reduction='batchmean') * args.unsup_loss_weight
    else:
        loss_unsup = None
    '''
    loss_l1 = 0
    numel = 0
    for param in model.parameters():
        loss_l1 += torch.sum(torch.abs(param))
        numel += param.nelement()
    #loss_l1 = loss_l1 * args.l1_reg / float(numel)
    loss_l1 = loss_l1 * args.l1_reg
    '''
    all_linear1_params = torch.cat([x.view(-1) for x in model.parameters()])
    loss_l1 = args.l1_reg * torch.norm(all_linear1_params, 1)

    #loss = loss_sup + loss_l1

    optimizer.zero_grad()
    loss_sup.backward()
    loss_l1.backward()
    if loss_unsup is not None:
        loss_unsup.backward()
    optimizer.step()

    top1.update( prec1.item(), input.size(0) )
    losses_sup.update( loss_sup.data.item(), input.size(0))
    losses_l1.update( loss_l1.data.item(), input.size(0))
    if loss_unsup is not None:
        losses_unsup.update( loss_unsup.data.item(), input.size(0))
    batch_time.update(time.time()-t)

def test(model, val_loader):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True).long()
        input = input.cuda()
        global zca
        input = zca( global_contrast_normalize( input ) )
        input_var  = input
        target_var = target

        # compute output
        with torch.no_grad():
            output = model(input_var)
        loss = torch.nn.functional.cross_entropy(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, it):
    lr = args.lr * ( args.lr_decay_rate ** int(it >= args.lr_decay_at) )
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def initialize_loader():
    global dataset_train, dataset_test, args
    train_loader    = torch.utils.data.DataLoader( dataset_train,   batch_size=args.batch_size, shuffle=True,   num_workers=args.num_workers, pin_memory=True )
    test_loader     = torch.utils.data.DataLoader( dataset_test,    batch_size=args.batch_size, shuffle=False,  num_workers=5, pin_memory=True )
    return train_loader, test_loader

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    global exp_dir
    filename = os.path.join(exp_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join( exp_dir, 'model_best.pth.tar') )

if __name__ == '__main__':
    main()
