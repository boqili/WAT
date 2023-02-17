import os
import argparse
import torchvision
import torch.optim as optim
from torchvision import transforms
import datetime
from models import *
import numpy as np
import attack_generator as attack
from data_loader import get_cifar10_loader, get_cifar100_loader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time

parser = argparse.ArgumentParser(description='PyTorch Worst-class Adversarial Training ')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--epsilon', type=float, default=0.031, help='perturbation bound')
parser.add_argument('--num_steps', type=int, default=10, help='maximum perturbation step K')
parser.add_argument('--step_size', type=float, default=0.007, help='step size')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--net', type=str, default="resnet18",help="decide which network to use,choose from resnet18,WRN")
parser.add_argument('--beta',type=float,default=6.0,help='regularization parameter')
parser.add_argument('--batch_size',type=int,default=128,help='batch size for data_loader')
parser.add_argument('--dataset', type=str, default="cifar10", help="choose from cifar10, cifar100")
parser.add_argument('--rand_init', type=bool, default=True, help="whether to initialize adversarial sample with random noise")

parser.add_argument('--depth', type=int, default=34, help='WRN depth')
parser.add_argument('--width_factor', type=int, default=10, help='WRN width factor') 
parser.add_argument('--drop_rate', type=float, default=0.0, help='WRN drop rate')

parser.add_argument('--out_dir',type=str,default='./results/',help='dir of output')
parser.add_argument('--resume', type=str, default='', help='whether to resume training')
parser.add_argument('--alg', type=str, default='none', help='use which algorithm')
parser.add_argument('--eta', type=float, default=0.1, help='hyper-parameter used in WAT')
parser.add_argument('--class_num', type=int, default=10, help='class number')
args = parser.parse_args()

# settings
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


args.out_dir = args.out_dir + '{}/'.format(args.alg)
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

def TRADES_loss(adv_logits, natural_logits, target, beta):
    # Based on the repo of TREADES: https://github.com/yaodongyu/TRADES
    batch_size = len(target)
    criterion_kl = nn.KLDivLoss(size_average=False).cuda()
    loss_natural = nn.CrossEntropyLoss(reduction='mean')(natural_logits, target)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(adv_logits, dim=1),
                                                         F.softmax(natural_logits, dim=1))
    loss = loss_natural + beta * loss_robust
    return loss

def TRADES_classwise_loss(adv_logits, natural_logits, target):
    criterion_kl = nn.KLDivLoss(reduction='none').cuda()
    loss_natural = nn.CrossEntropyLoss(reduction='none')(natural_logits, target)
    loss_robust = criterion_kl(F.log_softmax(adv_logits, dim=1),
                                                         F.softmax(natural_logits, dim=1))
    return loss_natural, loss_robust.sum(dim=1)


def train(model, train_loader, optimizer, epoch):
    global iter_num
    global nat_class_weights
    global bndy_class_weights

    starttime = datetime.datetime.now()
    loss_sum = 0
    train_iter = 0
    for data, target in train_loader:
        train_iter+=1
        iter_num+=1
        data, target = data.cuda(), target.cuda()
        output_adv = attack.trades(model,data,target, epsilon=args.epsilon, step_size=args.step_size, num_steps=args.num_steps,
                                        loss_fn='trades', category='trades', rand_init=True)
        # set model.train after be attacked
        model.train()
        optimizer.zero_grad()

        natural_logits = model(data)
        adv_logits = model(output_adv)

        iter_nat_loss, iter_bndy_loss = TRADES_classwise_loss(adv_logits,natural_logits,target)

        for i in range(args.class_num):
            if i == 0:
                nat_loss = iter_nat_loss[target == i].sum() * nat_class_weights[i]
                bndy_loss = iter_bndy_loss[target == i].sum() * bndy_class_weights[i]
            else:
                nat_loss += iter_nat_loss[target == i].sum() * nat_class_weights[i]
                bndy_loss += iter_bndy_loss[target == i].sum() * bndy_class_weights[i]
        loss = (nat_loss + args.beta * bndy_loss) / data.shape[0]
        
        # In WAT, we add excepted training loss in the decision set.
        if args.alg == 'wat': # our method
            loss += TRADES_loss(adv_logits, natural_logits, target, args.beta) * bndy_class_weights[args.class_num]

        loss_sum += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds

    return time, loss_sum

def validate(model, valid_loader, wat_valid_nat_cost, wat_valid_bndy_cost):
    val_iter = 0
    model.eval()
    epoch_val_bndy_loss = torch.zeros(args.class_num+1).cuda()
    epoch_val_bndy_loss.requires_grad = False
    epoch_val_nat_loss = torch.zeros(args.class_num+1).cuda()
    epoch_val_nat_loss.requires_grad = False
    correct = 0
    val_class_wise_acc = []
    val_class_wise_num = []
    for i in range(args.class_num):
        val_class_wise_acc.append(0)
        val_class_wise_num.append(0)
    model.zero_grad()
    for data, target in valid_loader:
        val_iter+=1
        data, target = data.cuda(), target.cuda()

        output_adv = attack.trades(model, data, target, args.epsilon, args.step_size, args.num_steps,"trades","trades",True)
        
        with torch.no_grad():
            natural_logits = model(data)
            adv_logits = model(output_adv)
            iter_nat_loss, iter_bndy_loss = TRADES_classwise_loss(adv_logits,natural_logits,target)
            pred = adv_logits.max(1, keepdim=True)[1] 
            target_view = target.view_as(pred)
            eq_mask = pred.eq(target_view) # equal mask
            # class-wise
            for i in range(args.class_num):
                label_mask = target_view == i
                val_class_wise_num[i] += label_mask.sum().item()
                val_class_wise_acc[i] += (eq_mask * label_mask).sum().item()
                epoch_val_nat_loss[i] += iter_nat_loss[target == i].sum()
                epoch_val_bndy_loss[i] += iter_bndy_loss[target == i].sum()
            epoch_val_nat_loss[args.class_num] += iter_nat_loss.sum() / args.class_num
            epoch_val_bndy_loss[args.class_num] += iter_bndy_loss.sum() / args.class_num
            
            correct += pred.eq(target.view_as(pred)).sum().item()
        
    wat_valid_nat_cost[epoch] = epoch_val_nat_loss / (val_iter * args.batch_size) * args.class_num
    wat_valid_bndy_cost[epoch] = epoch_val_bndy_loss / (val_iter * args.batch_size) * args.class_num

    val_acc = correct / len(valid_loader.dataset)
    for i in range(args.class_num):
        val_class_wise_acc[i] /= val_class_wise_num[i]
    model.zero_grad()
    return wat_valid_nat_cost, wat_valid_bndy_cost, val_acc, val_class_wise_acc

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    # The same as TRADES used in CIFAR-10
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, checkpoint=args.out_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

print('==> Load Test Data')
if args.dataset == "cifar10":
    train_loader, valid_loader, test_loader = get_cifar10_loader(args.batch_size)
    args.class_num = 10
if args.dataset == "cifar100":
    train_loader, valid_loader, test_loader = get_cifar100_loader(args.batch_size)
    args.class_num = 100

print('==> Load Model')
if args.net == "smallcnn":
    model = SmallCNN().cuda()
    net = "smallcnn"
elif args.net == "resnet18":
    model = ResNet18(args.class_num).cuda()
    net = "resnet18"
elif args.net == "WRN":
    model = Wide_ResNet(depth=args.depth, num_classes=args.class_num, widen_factor=args.width_factor, dropRate=args.drop_rate).cuda()
    net = "WRN{}-{}".format(args.depth, args.width_factor)
    
nat_class_weights = torch.ones(args.class_num+1).cuda()
bndy_class_weights = torch.ones(args.class_num+1).cuda()

model = torch.nn.DataParallel(model)
print(net)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

start_epoch = 0

param_list = '{}-{}-{}-{}-{}-{}-{}-{}'.format(args.lr,args.epochs,args.beta,args.epsilon,args.eta,net,args.dataset,datetime.datetime.now().strftime("%I%M%p-%b%d"))
if not os.path.exists(args.out_dir +'/ckpt/'+ param_list):
    os.makedirs(args.out_dir +'/ckpt/'+ param_list)
writer = SummaryWriter(log_dir=args.out_dir +'/visual/{}'.format(param_list))
iter_num = 0

if args.resume:
    print ('==> Adversarial Training Resuming from checkpoint ..')
    print(args.resume)
    assert os.path.isfile(args.resume)
    args.out_dir = os.path.dirname(args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
else:
    print('==> Worst-class Adversarial Training')

wat_valid_nat_cost = torch.zeros(args.epochs,args.class_num+1).cuda()
wat_valid_bndy_cost = torch.zeros(args.epochs,args.class_num+1).cuda()
wat_valid_cost = torch.zeros(args.epochs,args.class_num+1).cuda()

for epoch in tqdm(range(start_epoch, args.epochs)):
    train_time = 0
    train_loss = 0
    adjust_learning_rate(optimizer, epoch + 1)
    train_time, train_loss = train(model, train_loader, optimizer, epoch)
    if args.alg == 'wat': # our method
        wat_valid_nat_cost, wat_valid_bndy_cost, val_rob_acc, val_rob_class_wise_acc = validate(model, valid_loader, wat_valid_nat_cost, wat_valid_bndy_cost)
        for i in range(args.class_num+1):
            wat_valid_cost[epoch,i] = wat_valid_nat_cost[epoch,i] + args.beta * wat_valid_bndy_cost[epoch,i]
            class_factor = (torch.sum(wat_valid_cost,dim=0)* args.eta ).exp()
            nat_class_weights = args.class_num * class_factor / class_factor.sum()
            bndy_class_weights = args.class_num * class_factor / class_factor.sum()
    else: # vanilla TRADES
        wat_valid_nat_cost, wat_valid_bndy_cost, val_rob_acc, val_rob_class_wise_acc = validate(model, valid_loader, wat_valid_nat_cost, wat_valid_bndy_cost)
    ## Evalution
    nat_loss, test_nat_acc, nat_class_wise_acc = attack.eval_clean(model, test_loader, class_num=args.class_num)
    pgd100_loss, test_pgd100_acc, pgd100_class_wise_acc = attack.eval_robust(model,test_loader, perturb_steps=100, epsilon=0.031, step_size=0.003,loss_fn="cent",category="Madry",rand_init=True, class_num=args.class_num)
    cw_loss, test_cw_acc, cw_class_wise_acc = attack.eval_robust(model, test_loader, perturb_steps=30, epsilon=0.031, step_size=0.003,loss_fn="cw", category="Madry", rand_init=True, class_num=args.class_num)

    # print results
    print(
        'Epoch: [%d | %d] | Train Time: %.2f s | Natural Test Acc %.2f | PGD100 Test Acc %.2f | CW Test Acc %.2f |\n' % (
            epoch + 1,
            args.epochs,
            train_time,
            test_nat_acc,
            test_pgd100_acc,
            test_cw_acc)
    )
    # use tensorboard to record infos 
    for i in range(args.class_num):
        writer.add_scalar('nat/nat_class_wise_acc-{}'.format(i), nat_class_wise_acc[i], epoch + 1)
        writer.add_scalar('val/val_rob_class_wise_acc-{}'.format(i), val_rob_class_wise_acc[i], epoch + 1)
        writer.add_scalar('weight/nat-class-{}'.format(i), nat_class_weights[i].item(), epoch + 1)
        writer.add_scalar('weight/bndy-class-{}'.format(i), bndy_class_weights[i].item(), epoch + 1)

        writer.add_scalar('pgd/pgd100_class_wise_acc-{}'.format(i), pgd100_class_wise_acc[i], epoch + 1)
        writer.add_scalar('cw/cw_class_wise_acc-{}'.format(i), cw_class_wise_acc[i], epoch + 1)

    writer.add_scalar('worst/worst_nat_acc', min(nat_class_wise_acc), epoch + 1)
    writer.add_scalar('worst/worst_pgd100_acc', min(pgd100_class_wise_acc), epoch + 1)
    writer.add_scalar('worst/worst_cw_acc', min(cw_class_wise_acc), epoch + 1)

    writer.add_scalar('acc/test_nat_acc', test_nat_acc, epoch + 1)
    writer.add_scalar('acc/val_pgd100_acc', val_rob_acc, epoch + 1)
    writer.add_scalar('acc/test_pgd100_acc', test_pgd100_acc, epoch + 1)
    writer.add_scalar('acc/test_cw_acc', test_cw_acc, epoch + 1)

    writer.add_scalar('loss/train_loss', train_loss/len(train_loader.dataset), epoch + 1)
    writer.add_scalar('loss/nat_loss', nat_loss, epoch + 1)
    writer.add_scalar('loss/pgd100_loss', pgd100_loss, epoch + 1)
    writer.add_scalar('loss/cw_loss',cw_loss, epoch + 1)

    if (epoch + 1) % 3 == 0:
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        },checkpoint=args.out_dir +'/ckpt/'+ param_list,filename=str(epoch+1)+'.pt')
