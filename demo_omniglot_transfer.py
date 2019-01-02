import os.path
import argparse
from demo import run, get_args

parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', type=int, default=0,
                    help="Negative value means cpu-only")
parser.add_argument('--loss', type=str, default='MCL', choices=['KCL', 'MCL'],
                    help="The clustering criteria. Default: MCL")
parser.add_argument('--num_cluster', type=int, default=100,
                    help="The number of cluster. Default: 100 (unknown number of cluster)")
config = parser.parse_args()


print('STEP1: Train SPN on Omniglot background set')
if not os.path.isfile('outputs/Omniglot_VGGS_DPS.model.pth'):
    argv = '--loss DPS --dataset Omniglot --model_type vgg --model_name VGGS --schedule 30 40 --epochs 50'.split(' ')
    run(get_args(argv))
print('STEP1: Done')


omniglot_evaluation_alphabet_set = [
    'Angelic',
    'Atemayar_Qelisayer',
    'Atlantean',
    'Aurek',
    'Avesta',
    'Ge_ez',
    'Glagolitic',
    'Gurmukhi',
    'Kannada',
    'Keble',
    'Malayalam',
    'Manipuri',
    'Mongolian',
    'Old_Church_Slavonic',
    'Oriya',
    'Sylheti',
    'Syriac',
    'Tengwar',
    'Tibetan',
    'ULOG'
    ]

acc = {}
for i,alphabet in enumerate(omniglot_evaluation_alphabet_set):
    print('STEP2 [%d/20]: Clustering on Omniglot evaluation alphabet %s'%(i+1,alphabet))
    argv = '--gpuid %d --dataset Omniglot_eval_%s --model_type vgg --model_name VGGS --schedule 100 --epochs 150 --print_freq 0 --loss %s --out_dim %d --skip_eval --use_SPN'%(config.gpuid,alphabet,config.loss,config.num_cluster)
    argv = argv.split(' ')
    acc[alphabet] = run(get_args(argv))

print('ACC for all alphabets:',acc)
print('Average:',sum(acc.values()) / float(len(acc)))