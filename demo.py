import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import MultiStepLR

from learners.classification import Learner_Classification
from learners.clustering import Learner_Clustering
from learners.similarity import Learner_DensePairSimilarity
from utils.metric import Confusion, Timer, AverageMeter
from modules.pairwise import Class2Simi
import modules.criterion


def prepare_task_target(input, target, args):
    # Prepare the target for different criterion/tasks
    if args.loss == 'CE':  # For standard classification
        train_target = eval_target = target
    elif args.loss in ['KCL', 'MCL']:  # For clustering
        if args.use_SPN:  # For unsupervised clustering
            # Feed the input to SPN to get predictions
            _, train_target = args.SPN(input).max(1)  # Binaries the predictions
            train_target = train_target.float()
            train_target[train_target==0] = -1  # Simi:1, Dissimi:-1
        else:  # For supervised clustering
            # Convert class labels to pairwise similarity
            train_target = Class2Simi(target, mode='hinge')
        eval_target = target
    elif args.loss == 'DPS':  # For learning the SPN
        train_target = eval_target = Class2Simi(target, mode='cls')
    else:
        assert False,'Unsupported loss:'+args.loss
    return train_target.detach(), eval_target.detach()  # Make sure no gradients


def train(epoch, train_loader, learner, args):
    # This function optimize the objective

    # Initialize all meters
    data_timer = Timer()
    batch_timer = Timer()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    confusion = Confusion(args.out_dim)

    # Setup learner's configuration
    print('\n\n==== Epoch:{0} ===='.format(epoch))
    learner.train()
    learner.step_schedule(epoch)

    # The optimization loop
    data_timer.tic()
    batch_timer.tic()
    if args.print_freq>0:  # Enable to print mini-log
        print('Itr            |Batch time     |Data Time      |Loss')
    for i, (input, target) in enumerate(train_loader):

        data_time.update(data_timer.toc())  # measure data loading time

        # Prepare the inputs
        if args.use_gpu:
            input = input.cuda()
            target = target.cuda()
        train_target, eval_target = prepare_task_target(input, target, args)

        # Optimization
        loss, output  = learner.learn(input, train_target)

        # Update the performance meter
        confusion.add(output, eval_target)

        # Measure elapsed time
        batch_time.update(batch_timer.toc())
        data_timer.toc()

        # Mini-Logs
        losses.update(loss, input.size(0))
        if args.print_freq>0 and ((i%args.print_freq==0) or (i==len(train_loader)-1)):
            print('[{0:6d}/{1:6d}]\t'
                  '{batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                  '{data_time.val:.4f} ({data_time.avg:.4f})\t'
                  '{loss.val:.3f} ({loss.avg:.3f})'.format(
                i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

    # Loss-specific information
    if args.loss=='CE':
        print('[Train] ACC: ', confusion.acc())
    elif args.loss in ['KCL','MCL']:
        args.cluster2Class = confusion.optimal_assignment(train_loader.num_classes)  # Save the mapping in args to use in eval
        if args.out_dim <= 20:  # Avoid to print a large confusion matrix
            confusion.show()
        print('Clustering scores:', confusion.clusterscores())
        print('[Train] ACC: ', confusion.acc())
    elif args.loss=='DPS':
        confusion.show(width=15,row_labels=['GT_dis-simi','GT_simi'],column_labels=['Pred_dis-simi','Pred_simi'])
        print('[Train] similar pair f1-score:', confusion.f1score(1))  # f1-score for similar pair (label:1)
        print('[Train] dissimilar pair f1-score:', confusion.f1score(0))


def evaluate(eval_loader, model, args):

    # Initialize all meters
    confusion = Confusion(args.out_dim)

    print('---- Evaluation ----')
    model.eval()
    for i, (input, target) in enumerate(eval_loader):

        # Prepare the inputs
        if args.use_gpu:
            with torch.no_grad():
                input = input.cuda()
                target = target.cuda()
        _, eval_target = prepare_task_target(input, target, args)

        # Inference
        output = model(input)

        # Update the performance meter
        output = output.detach()
        confusion.add(output,eval_target)

    # Loss-specific information
    KPI = 0
    if args.loss == 'CE':
        KPI = confusion.acc()
        print('[Test] ACC: ', KPI)
    elif args.loss in ['KCL', 'MCL']:
        confusion.optimal_assignment(eval_loader.num_classes, args.cluster2Class)
        if args.out_dim<=20:
            confusion.show()
        print('Clustering scores:',confusion.clusterscores())
        KPI = confusion.acc()
        print('[Test] ACC: ', KPI)
    elif args.loss == 'DPS':
        confusion.show(width=15, row_labels=['GT_dis-simi', 'GT_simi'], column_labels=['Pred_dis-simi', 'Pred_simi'])
        KPI = confusion.f1score(1)
        print('[Test] similar pair f1-score:', KPI)  # f1-score for similar pair (label:1)
        print('[Test] dissimilar pair f1-score:', confusion.f1score(0))
    return KPI


def run(args):
    if not os.path.exists('outputs'):
        os.mkdir('outputs')

    # Select the optimization criterion/task
    if args.loss=='CE':
        # Classification
        LearnerClass = Learner_Classification
        criterion = nn.CrossEntropyLoss()
    elif args.loss in ['KCL', 'MCL']:
        # Clustering
        LearnerClass = Learner_Clustering
        criterion = modules.criterion.__dict__[args.loss]()
    elif args.loss=='DPS':
        # Dense-Pair Similarity Learning
        LearnerClass = Learner_DensePairSimilarity
        criterion = nn.CrossEntropyLoss()
        args.out_dim = 2  # force it

    # Prepare dataloaders
    loaderFuncs = __import__('dataloaders.'+args.dataset_type)
    loaderFuncs = loaderFuncs.__dict__[args.dataset_type]
    train_loader, eval_loader = loaderFuncs.__dict__[args.dataset](args.batch_size, args.workers)

    # Prepare the model
    if args.out_dim<0:  # Use ground-truth number of classes/clusters
        args.out_dim = train_loader.num_classes
    model = LearnerClass.create_model(args.model_type,args.model_name,args.out_dim)

    # Load pre-trained model
    if args.pretrained_model != '':  # Load model weights only
        print('=> Load model weights:', args.pretrained_model)
        model_state = torch.load(args.pretrained_model,
                                 map_location=lambda storage, loc: storage)  # Load to CPU as the default!
        model.load_state_dict(model_state, strict=args.strict)
        print('=> Load Done')

    # Load the pre-trained Similarity Prediction Network (SPN, or the G function in paper)
    if args.use_SPN:
        # To load a custom SPN, you can modify here.
        SPN = Learner_DensePairSimilarity.create_model(args.SPN_model_type, args.SPN_model_name, 2)
        print('=> Load SPN model weights:', args.SPN_pretrained_model)
        SPN_state = torch.load(args.SPN_pretrained_model,
                                 map_location=lambda storage, loc: storage)  # Load to CPU as the default!
        SPN.load_state_dict(SPN_state)
        print('=> Load SPN Done')
        print('SPN model:', SPN)
        #SPN.eval()  # Tips: Stay in train mode, so the BN layers of SPN adapt to the new domain
        args.SPN = SPN  # It will be used in prepare_task_target()

    # GPU
    if args.use_gpu:
        torch.cuda.set_device(args.gpuid[0])
        cudnn.benchmark = True  # make it train faster
        model = model.cuda()
        criterion = criterion.cuda()
        if args.SPN is not None:
            args.SPN = args.SPN.cuda()

    # Multi-GPU
    if len(args.gpuid) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpuid, output_device=args.gpuid[0])

    print('Main model:',model)
    print('Criterion:', criterion)

    # Evaluation Only
    if args.skip_train:
        cudnn.benchmark = False  # save warm-up time
        eval_loader = eval_loader if eval_loader is not None else train_loader
        KPI = evaluate(eval_loader, model, args)
        return KPI

    # Prepare the learner
    optim_args = {'lr':args.lr}
    if args.optimizer=='SGD':
        optim_args['momentum'] = 0.9
    optimizer = torch.optim.__dict__[args.optimizer](model.parameters(), **optim_args)
    scheduler = MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)
    learner = LearnerClass(model, criterion, optimizer, scheduler)

    # Start optimization
    if args.resume:
        args.start_epoch = learner.resume(args.resume) + 1  # Start from next epoch
    KPI = 0
    for epoch in range(args.start_epoch, args.epochs):
        train(epoch, train_loader, learner, args)
        if eval_loader is not None and ((not args.skip_eval) or (epoch==args.epochs-1)):
            KPI = evaluate(eval_loader, model, args)
        # Save checkpoint at each LR steps and the end of optimization
        if epoch+1 in args.schedule+[args.epochs]:
            learner.snapshot("outputs/%s_%s_%s"%(args.dataset, args.model_name, args.saveid), KPI)
    return KPI


def get_args(argv):
    # This function prepares the variables shared across demo.py

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--model_type', type=str, default='lenet', help="lenet(default)|vgg|resnet")
    parser.add_argument('--model_name', type=str, default='LeNet', help="LeNet(default)|LeNetC|VGGS|VGG8|VGG16|ResNet18|ResNet101 ...")
    parser.add_argument('--dataset_type', type=str, default='default')
    parser.add_argument('--dataset', type=str, default='MNIST', help="MNIST(default)|CIFAR10|CIFAR100|Omniglot|Omniglot_eval_Old_Church_Slavonic ...")
    parser.add_argument('--out_dim', type=int, default=-1,
                        help="Output dimension of network. Default:-1 (Use ground-truth)")
    parser.add_argument('--workers', type=int, default=2, help="#Thread for dataloader")
    parser.add_argument('--epochs', type=int, default=30, help="End epoch")
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--loss', type=str, default='MCL', choices=['CE', 'KCL', 'MCL', 'DPS'],
                        help="CE(cross-entropy)|KCL|MCL(default)|DPS(Dense-Pair Similarity)")
    parser.add_argument('--schedule', nargs="+", type=int, default=[10, 20],
                        help="The list of epoch numbers to reduce learning rate by factor of 0.1")
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--print_freq', type=float, default=100, help="Print the log at every x iteration")
    parser.add_argument('--resume', type=str, default='', help="The path to checkpoint file (*.checkpoint.pth)")
    parser.add_argument('--pretrained_model', type=str, default='',
                        help="The path to model file (*.best_model.pth). Do NOT use checkpoint file here.")
    parser.add_argument('--saveid', type=str, default='', help="The appendix to the saved model")
    parser.add_argument('--skip_train', dest='skip_train', default=False, action='store_true', help="Evaluation only")
    parser.add_argument('--skip_eval', dest='skip_eval', default=False, action='store_true', help="Only do the evaluation after training is done")
    parser.add_argument('--no-strict', dest='strict', default=True, action='store_false',
                        help="The pretrained state dict doesn't need to fit the model")

    # For SPN
    parser.add_argument('--use_SPN', dest='use_SPN', default=False, action='store_true',
                        help="Use Similarity Prediction Network")
    parser.add_argument('--SPN_model_type', type=str, default='vgg', help="This option is only valid when use_SPN=True")
    parser.add_argument('--SPN_model_name', type=str, default='VGGS', help="This option is only valid when use_SPN=True")
    parser.add_argument('--SPN_pretrained_model', type=str, default='outputs/Omniglot_VGGS_DPS.model.pth', help="This option is only valid when use_SPN=True")

    args = parser.parse_args(argv)

    # Initialize some useful flags
    args.use_gpu = args.gpuid[0] >= 0
    args.start_epoch = 0
    args.saveid = args.loss if args.saveid == '' else args.saveid
    args.cluster2Class = None
    args.SPN = None

    return args

if __name__ == '__main__':
    run(get_args(sys.argv[1:]))
