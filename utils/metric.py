from __future__ import print_function
import time
import torch
from scipy.optimize import linear_sum_assignment as hungarian
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score


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
        self.avg = float(self.sum) / self.count


class Timer(object):
    """
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.interval = 0
        self.time = time.time()

    def value(self):
        return time.time() - self.time
        
    def tic(self):
        self.time = time.time()
        
    def toc(self):
        self.interval = time.time() - self.time
        self.time = time.time()
        return self.interval


class Confusion(object):
    """
    column of confusion matrix: predicted index
    row of confusion matrix: target index
    """
    def __init__(self, k, normalized = False):
        super(Confusion, self).__init__()
        self.k = k
        self.conf = torch.LongTensor(k,k)
        self.normalized = normalized
        self.reset()

    def reset(self):
        self.conf.fill_(0)
        self.gt_n_cluster = None

    def cuda(self):
        self.conf = self.conf.cuda()

    def add(self, output, target):
        output = output.squeeze()
        target = target.squeeze()
        assert output.size(0) == target.size(0), \
                'number of targets and outputs do not match'
        if output.ndimension()>1: #it is the raw probabilities over classes
            assert output.size(1) == self.conf.size(0), \
                'number of outputs does not match size of confusion matrix'
        
            _,pred = output.max(1) #find the predicted class
        else: #it is already the predicted class
            pred = output
        indices = (target*self.conf.stride(0) + pred.squeeze_().type_as(target)).type_as(self.conf)
        ones = torch.ones(1).type_as(self.conf).expand(indices.size(0))
        self._conf_flat = self.conf.view(-1)
        self._conf_flat.index_add_(0, indices, ones)

    def classIoU(self,ignore_last=False):
        confusion_tensor = self.conf
        if ignore_last:
            confusion_tensor = self.conf.narrow(0,0,self.k-1).narrow(1,0,self.k-1)
        union = confusion_tensor.sum(0).view(-1) + confusion_tensor.sum(1).view(-1) - confusion_tensor.diag().view(-1)
        acc = confusion_tensor.diag().float().view(-1).div(union.float()+1)
        return acc
        
    def recall(self,clsId):
        i = clsId
        TP = self.conf[i,i].sum().item()
        TPuFN = self.conf[i,:].sum().item()
        if TPuFN==0:
            return 0
        return float(TP)/TPuFN
        
    def precision(self,clsId):
        i = clsId
        TP = self.conf[i,i].sum().item()
        TPuFP = self.conf[:,i].sum().item()
        if TPuFP==0:
            return 0
        return float(TP)/TPuFP
        
    def f1score(self,clsId):
        r = self.recall(clsId)
        p = self.precision(clsId)
        if (p+r)==0:
            return 0
        return 2*float(p*r)/(p+r)
        
    def acc(self):
        TP = self.conf.diag().sum().item()
        total = self.conf.sum().item()
        if total==0:
            return 0
        return float(TP)/total
        
    def optimal_assignment(self,gt_n_cluster=None,assign=None):
        if assign is None:
            mat = -self.conf.cpu().numpy() #hungaian finds the minimum cost
            r,assign = hungarian(mat)
        self.conf = self.conf[:,assign]
        self.gt_n_cluster = gt_n_cluster
        return assign
        
    def show(self,width=6,row_labels=None,column_labels=None):
        print("Confusion Matrix:")
        conf = self.conf
        rows = self.gt_n_cluster or conf.size(0)
        cols = conf.size(1)
        if column_labels is not None:
            print(("%" + str(width) + "s") % '', end='')
            for c in column_labels:
                print(("%" + str(width) + "s") % c, end='')
            print('')
        for i in range(0,rows):
            if row_labels is not None:
                print(("%" + str(width) + "s|") % row_labels[i], end='')
            for j in range(0,cols):
                print(("%"+str(width)+".d")%conf[i,j],end='')
            print('')
        
    def conf2label(self):
        conf=self.conf
        gt_classes_count=conf.sum(1).squeeze()
        n_sample = gt_classes_count.sum().item()
        gt_label = torch.zeros(n_sample)
        pred_label = torch.zeros(n_sample)
        cur_idx = 0
        for c in range(conf.size(0)):
            if gt_classes_count[c]>0:
                gt_label[cur_idx:cur_idx+gt_classes_count[c]].fill_(c)
            for p in range(conf.size(1)):
                if conf[c][p]>0:
                    pred_label[cur_idx:cur_idx+conf[c][p]].fill_(p)
                cur_idx = cur_idx + conf[c][p];
        return gt_label,pred_label
    
    def clusterscores(self):
        target,pred = self.conf2label()
        NMI = normalized_mutual_info_score(target,pred)
        ARI = adjusted_rand_score(target,pred)
        AMI = adjusted_mutual_info_score(target,pred)
        return {'NMI':NMI,'ARI':ARI,'AMI':AMI}
