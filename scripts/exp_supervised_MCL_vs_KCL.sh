# MNIST @ LeNet
python demo.py --loss MCL
python demo.py --loss KCL
python demo.py --loss CE

# CIFAR10 @ VGG8
python demo.py --dataset CIFAR10 --model_type vgg --model_name VGG8 --schedule 80 120 --epochs 140 --loss MCL
python demo.py --dataset CIFAR10 --model_type vgg --model_name VGG8 --schedule 80 120 --epochs 140 --loss KCL
python demo.py --dataset CIFAR10 --model_type vgg --model_name VGG8 --schedule 80 120 --epochs 140 --loss CE

# CIFAR10 @ VGG16
python demo.py --dataset CIFAR10 --model_type vgg --model_name VGG16 --schedule 80 120 --epochs 140 --loss MCL
python demo.py --dataset CIFAR10 --model_type vgg --model_name VGG16 --schedule 80 120 --epochs 140 --loss KCL
python demo.py --dataset CIFAR10 --model_type vgg --model_name VGG16 --schedule 80 120 --epochs 140 --loss CE

# CIFAR10 @ ResNet101
python demo.py --dataset CIFAR10 --model_type resnet --model_name ResNet101 --schedule 80 120 --epochs 140 --loss MCL --optimizer SGD --lr 0.1
python demo.py --dataset CIFAR10 --model_type resnet --model_name ResNet101 --schedule 80 120 --epochs 140 --loss KCL --optimizer SGD --lr 0.1
python demo.py --dataset CIFAR10 --model_type resnet --model_name ResNet101 --schedule 80 120 --epochs 140 --loss CE  --optimizer SGD --lr 0.1

# CIFAR100
python demo.py --dataset CIFAR100 --model_type vgg --model_name VGG8 --batch_size 1000 --schedule 100 150 --epochs 180 --loss MCL
python demo.py --dataset CIFAR100 --model_type vgg --model_name VGG8 --batch_size 1000 --schedule 100 150 --epochs 180 --loss KCL
python demo.py --dataset CIFAR100 --model_type vgg --model_name VGG8 --batch_size 1000 --schedule 100 150 --epochs 180 --loss CE