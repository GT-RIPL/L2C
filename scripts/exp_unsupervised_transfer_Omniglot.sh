# Train SPN
python demo.py --loss DPS --dataset Omniglot --model_type vgg --model_name VGGS --schedule 30 40 --epochs 50
# Do clustering on one alphabet dataset
python demo.py --dataset Omniglot_eval_Old_Church_Slavonic --model_type vgg --model_name VGGS --schedule 100 --epochs 150 --use_SPN