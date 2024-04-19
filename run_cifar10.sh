
### wandb
# --wandb --project tta-defense-target 

# indiscriminate
CUDA_VISIBLE_DEVICES=0 nohup python test_attack.py --cfg ./cfgs/cifar10/tent.yaml \
MODEL.ARCH resnet26 MODEL.NORM bn ATTACK.STEPS 100 ATTACK.SOURCE 40 > /dev/null &
sleep 1

# target 
CUDA_VISIBLE_DEVICES=0 nohup python test_attack.py --cfg ./cfgs/cifar10/tent.yaml \
ATTACK.TARGETED True \
MODEL.ARCH resnet26 MODEL.NORM bn ATTACK.STEPS 100 ATTACK.SOURCE 40 > /dev/null &
sleep 1
