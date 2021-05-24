# this code is modified from the pytorch example code: https://github.com/pytorch/examples/blob/master/imagenet/main.py
# after the model is trained, you might use convert_model.py to remove the data parallel module to make the model as standalone weight.
#
# Bolei Zhou

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import wideresnet
import pdb
# import wandb
# wandb.login()
is_first = True
pred_stack=[]
target_stack=[]
correct_predicted_labels = torch.zeros(365,dtype=torch.float64)
total_labels = torch.zeros(365,dtype=torch.float64)
calculate_per_class_acc = False

# import nonechucks as nc

PATH = 'models/resnet18/'
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch Places365 Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_false',
                    help='use pre-trained model')
parser.add_argument('--num_classes',default=365, type=int, help='num of class in the model')
parser.add_argument('--dataset',default='places365',help='which dataset to train')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)
    print("try except in /nethome/mummettuguli3/anaconda2/envs/my_basic_env_3/lib/python3.6/site-packages/torchvision/datasets/folder.py")
    # wandb.init(project="places365_"+args.arch.lower(), config=args)
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch.lower().startswith('wideresnet'):
        # a customized resnet model with last feature map size as 14x14 for better class activation mapping
        model  = wideresnet.resnet50(num_classes=args.num_classes)
    else:
        model = models.__dict__[args.arch](num_classes=args.num_classes)

    if args.arch.lower().startswith('alexnet') or args.arch.lower().startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()
    print(model)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            # args.start_epoch = checkpoint['epoch']
            # best_prec1 = checkpoint['best_prec1']
            # model.load_state_dict(checkpoint['state_dict'])
            model.load_state_dict(checkpoint)
            # print("=> loaded checkpoint '{}' (epoch {})"
            #       .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    # train_dataset = nc.SafeDataset(train_dataset)
    
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    # val_dataset = nc.SafeDataset(val_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # train_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(traindir, transforms.Compose([
    #         transforms.RandomSizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=True)

    # val_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(valdir, transforms.Compose([
    #         transforms.Scale(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # PATH=os.path.join(PATH, args.arch.lower())

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        torch.save(model.state_dict(), PATH+'model_state_epoch_'+str(epoch+1)+'.pt')

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, args.arch.lower())


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input_var, target_var) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target_var = target_var.cuda(async=True)
        # input_var = torch.autograd.Variable(input)
        # target_var = torch.autograd.Variable(target)
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
        # pdb.set_trace()
        # losses.update(loss.data[0], input.size(0))
        # top1.update(prec1[0], input.size(0))
        # top5.update(prec5[0], input.size(0))
        losses.update(loss.data.item(), input_var.size(0))
        top1.update(prec1.item(), input_var.size(0))
        top5.update(prec5.item(), input_var.size(0))
        # wandb.log({'loss': loss.data.item(), 'top1': prec1.item(), 'top5':prec5.item()})

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    
    with torch.no_grad():
        end = time.time()
        for i, (input_var, target_var) in enumerate(val_loader):
            target_var = target_var.cuda(async=True)
            # input_var = torch.autograd.Variable(input, volatile=True)
            # target_var = torch.autograd.Variable(target, volatile=True)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            accuracy(output.data, target_var, topk=(1, 5))
            # measure accuracy and record loss
            # prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
            # # losses.update(loss.data[0], input.size(0))
            # # top1.update(prec1[0], input.size(0))
            # # top5.update(prec5[0], input.size(0))
            # losses.update(loss.data.item(), input_var.size(0))
            # top1.update(prec1.item(), input_var.size(0))
            # top5.update(prec5.item(), input_var.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))

    # return top1.avg
    correct = pred_stack.eq(target_stack)
    # print("correct shape:",correct.shape)
    res = []
    topk=(1, 5)
    for k in topk:
        # correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)

        # torch.bincount(target).cuda(args_.gpu, non_blocking=True) 
        if calculate_per_class_acc:
            for i in range(365):
                indices_of_occurance_of_i = target_stack[:k]==i
                # if k==1:
                #     indices_of_occurance_of_i = indices_of_occurance_of_i.squeeze()
                # num_occurance_of_i = torch.sum(indices_of_occurance_of_i).cuda(args_.gpu, non_blocking=True)
                num_occurance_of_i = torch.sum(indices_of_occurance_of_i)
                
                correct_k = correct[:k].contiguous()
                if k==1:
                    indices_of_occurance_of_i = indices_of_occurance_of_i.squeeze()
                    correct_k = correct_k.view(-1)
                correct_predicted_labels[i] += correct_k[indices_of_occurance_of_i].float().sum(0, keepdim=True).item()
                total_labels[i] += num_occurance_of_i
            
            per_class_accuracy = correct_predicted_labels/total_labels
            # print('per_class_top'+str(k)+'_accuracy_epoch'+args.resume.split("_")[3]+":",per_class_accuracy)
            torch.save(per_class_accuracy, 'models/resnet50/accuracy/per_class_top'+str(k)+'_accuracy_epoch'+args.resume.split("_")[3])
        
        overall_correct_preds = correct[:k].contiguous().float().sum()
        # pdb.set_trace()
        overall_accuracy = overall_correct_preds/correct[:k].shape[1]
        res.append(overall_accuracy.item())
        # pdb.set_trace()
    torch.save(res, 'models/resnet50/accuracy/overall/top'+str(topk[0])+'_top'+str(topk[1])+'_accuracy_epoch'+args.resume.split("_")[3])


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, PATH+filename + '_latest.pth.tar')
    if is_best:
        shutil.copyfile(PATH+filename + '_latest.pth.tar', PATH+filename + '_best.pth.tar')


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    global is_first
    global pred_stack
    global target_stack
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        
        target = target.cpu()
        pred = pred.cpu()

        target_expanded = target.view(1, -1).expand_as(pred)
        if is_first:
            pred_stack = pred
            target_stack = target_expanded
            is_first = False
        else:
            pred_stack = torch.hstack((pred_stack,pred))
            target_stack = torch.hstack((target_stack,target_expanded))
        
        # correct = pred.eq(target.view(1, -1).expand_as(pred))

        # res = []
        # for k in topk:
        #     # correct_k = correct[:k].view(-1).float().sum(0)
        #     correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        #     res.append(correct_k.mul_(100.0 / batch_size))
        # return res


if __name__ == '__main__':
    main()