# MedBN-robust-test-time-adpatation
[CVPR 2024] Official implementation of [MedBN: Robust Test Time Adaptation against Malicious Test Samples](https://arxiv.org/abs/2403.19326) by Hyejin Park\*, Jeongyeon Hwang\*, Sunung Mun, Sangdon Park, and Jungseul Ok

You can visit [our project page](http://hyejin-s.github.io/medbn) 🚀!

## Requirements
- [RobustBench](https://github.com/RobustBench/robustbench)

## Datasets
You need to download the CIFAR10-C and CIFAR100-C data to `../data/`

## Pretrained Model
We utilize a pre-trained model from the [DIA](https://github.com/inspire-group/tta_risk) repository for the CIFAR dataset, and from [RobustBench](https://github.com/RobustBench/robustbench) for the ImageNet dataset.

## Usage

Examples for running code for CIFAR10-C
```
python test_attack.py --cfg ./cfgs/cifar10/[method].yaml \
MODEL.ARCH resnet26 MODEL.NORM [bn/med] ATTACK.STEPS 100 ATTACK.SOURCE 40 
```


If you have any questions, please contact Hyejin Park by [parkebbi2@gmail.com]


## Citation
If our MedBN method are helpful in your research, please consider citing our paper:
```
@inproceedings{park2024medbn,
  title={MedBN: Robust Test-Time Adaptation against Malicious Test Samples},
  author={Park, Hyejin and Hwang, Jeongyeon and Mun, Sunung and Park, Sangdon and Ok, Jungseul},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5997--6007},
  year={2024}
}
```


## Acknowledgement
The code is inspired by [DIA](https://github.com/inspire-group/tta_risk), [TENT](https://github.com/DequanWang/tent), [EATA](https://github.com/mr-eggplant/EATA), [SAR](https://github.com/mr-eggplant/SAR/tree/main), and [SoTTA](https://github.com/taeckyung/SoTTA).
