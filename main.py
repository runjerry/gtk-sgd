'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

import os
import argparse
from tqdm import tqdm

from models import *
from utils import progress_bar, set_random_seed
from injective_sgd import iSGD
from slice_sgd import sSGD
from gtk_sgd import affineSGD


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# lr=0.1 for mlp with bias and 0.01 without bias
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model', default='mlp1', type=str)
parser.add_argument('--act', default='relu', type=str)
parser.add_argument('--bias', action='store_false')
parser.add_argument('--fullrank', action='store_true')
parser.add_argument('--fixedRV', action='store_true')
parser.add_argument('--weightonly', action='store_true')
parser.add_argument('--samenorm', action='store_true')
parser.add_argument('--norm', action='store_true')
parser.add_argument('--diag', action='store_true')
parser.add_argument('--exp', default=None, type=float)
parser.add_argument('--optimizer', default='sgd', type=str)
parser.add_argument('--log_dir', default='runs/cifar10', type=str)
parser.add_argument('--epoch', default=200, type=int)
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--renorm', default=None, type=str)
parser.add_argument('--option3', action='store_true')
parser.add_argument('--init', default=None, type=str)
parser.add_argument('--extra', default=None, type=str)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# set random seed
seed = set_random_seed(args.seed)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()

if args.model == 'mlp1':
    net = MLP(initializer=args.init, use_bias=args.bias, activation=args.act)
elif args.model == 'mlp2':
    net = MLP2(initializer=args.init, use_bias=args.bias)
elif args.model == 'mlp3':
    net = MLP3(initializer=args.init, use_bias=args.bias)
else:
    raise ValueError(f"{args.model} is not a supported model.")

net = net.to(device)
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# Training config
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
best_acc = 0  # best test accuracy
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print('==> Loaded checkpoint at epoch: %d, acc: %.2f%%' % (start_epoch, best_acc))

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr,
#                       momentum=0.9, weight_decay=5e-4)
if args.optimizer == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=args.lr)
elif args.optimizer == 'isgd':
    optimizer = iSGD(
        net.parameters(), lr=args.lr, option3=args.option3, renorm=args.renorm)
elif args.optimizer == 'ssgd':
    optimizer = sSGD(
        net.parameters(), lr=args.lr, option3=args.option3, renorm=args.renorm)
elif args.optimizer == 'gtk':
    optimizer = affineSGD(net.parameters(), lr=args.lr,
                          use_bias=args.bias, fullrank=args.fullrank,
                          fixed_rand_vec=args.fixedRV,
                          weight_only=args.weightonly,
                          same_norm=args.samenorm,
                          norm=args.norm,
                          diag=args.diag,
                          exponential=args.exp)
else:
    raise ValueError(f"{args.optimizer} optimizer is not supported.")
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Summary writter
if args.fullrank:
    str_fullrank = '_fullrank'
else:
    str_fullrank = ''
if args.fixedRV:
    str_rv = '_fixedRV'
else:
    str_rv = ''
if args.weightonly:
    str_wo = '_weightonly'
else:
    str_wo = ''
if args.samenorm:
    str_sn = '_samenorm'
else:
    str_sn = ''
if args.norm:
    str_norm = '_norm'
else:
    str_norm = ''
if args.diag:
    str_diag = '_diag'
else:
    str_diag = ''
if args.exp:
    str_exp = f'_exp{args.exp}'
else:
    str_exp = ''
if args.renorm is None:
    str_renorm = ''
else:
    str_renorm = '_renorm-' + args.renorm
if args.option3:
    str_option3 = '_option3'
else:
    str_option3 = ''
if args.extra is None:
    str_extra = ''
else:
    str_extra = '_' + args.extra
log_dir = os.path.join(
    args.log_dir, args.optimizer,
    '%s_%s_lr%.3f_epoch%d_seed%d_init-%s%s%s%s%s%s%s%s%s%s%s' % (
        args.model, args.act, args.lr, args.epoch, args.seed, args.init,
        str_fullrank, str_rv, str_wo, str_sn, str_norm, str_diag, str_exp,
        str_renorm, str_option3, str_extra))
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    writer.add_scalar('train/lr', lr_scheduler.get_last_lr()[0], epoch)
    desc = ('[%s][LR=%.4f] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (args.optimizer, lr_scheduler.get_last_lr()[0], 0, 0, correct, total))
    prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), 
                    desc=desc, leave=True)
    # for batch_idx, (inputs, targets) in enumerate(trainloader):
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        desc = ('[%s][LR=%.4f] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (args.optimizer, lr_scheduler.get_last_lr()[0], train_loss / (batch_idx + 1), 
                 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    writer.add_scalar('train/loss', train_loss/(batch_idx + 1), epoch)
    writer.add_scalar('train/acc', 100. * correct / total, epoch)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    desc = ('[%s][LR=%.4f] Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (args.optimizer, lr_scheduler.get_last_lr()[0], test_loss/(0+1), 0, correct, total))

    prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=True)
    with torch.no_grad():
        # for batch_idx, (inputs, targets) in enumerate(testloader):
        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            desc = ('[%s][LR=%.4f] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (args.optimizer, lr_scheduler.get_last_lr()[0], test_loss / (batch_idx + 1), 
                       100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100. * correct / total
    writer.add_scalar('test/loss', test_loss / (batch_idx + 1), epoch)
    writer.add_scalar('test/acc', acc, epoch)

    # Save checkpoint.
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


def main():
    for epoch in range(start_epoch, args.epoch):
        train(epoch)
        test(epoch)
        lr_scheduler.step()
    # writer.add_hparams(vars(args), metrics)
    writer.close()
    return best_acc


if __name__ == '__main__':
    main()
