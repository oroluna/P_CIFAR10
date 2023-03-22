# P_CIFAR10 HYDRA

Utilizando el Python 3.8.3

# P_CIFAR10 HYDRA

## Installation

### Install pip

```
sudo apt-get update
```

```
sudo apt-get install python-pip
```

### Install requirements

```
pip install -r requirements.txt 
```

### run experiments
```
python main.py --multirun 'model=choice(vgg11_train, resnet18_train, googlenet_train)' 'optimizer=choice(adam, sgd)'
```

### Dataset slice
- ds_small
- ds_medium
- ds_big
- ds_default
```
 python main.py data=ds_medium 
 ```
 
 ### Models

 ```
 python main.py model=vgg11_train (default)
 ```
 | model  | argument |
| ------------- | ------------- |
| VGG11 (Deault)  | vgg11_train  |
| DenseNet161  | densenet161_train  |
| DenseNet169  | densenet169_train  |
| DenseNet201  | densenet201_train  |
| densenet_cifar  | densenetcifar_train  |
| DLA  | dla_train  |
| DPN26  | dpn26_train  |
| DPN92  | dpn92_train  |
| EfficientNetB0  | efficientnetb0_train  |
| GoogLeNet  | googlenet_train  |
| MobileNet  | mobilenet_train  |
| MobileNetV2  | mobilenetv2_train  |
| PNASNetA  | pnasneta_train  |
| PNASNetB  | pnasnetb_train  |
| PreActResNet18  | preactresnet18_train  |
| PreActResNet34  | preactresnet34_train  |
| PreActResNet50  | preactresnet50_train  |
| PreActResNet101  | preactresnet101_train  |
| PreActResNet152  | preactresnet152_train  |
| RegNetX_200MF  | regnetx200mf_train  |
| RegNetX_400MF  | regnetx400mf_train  |
| RegNetY_400MF  | regnety400mf_train  |
| ResNet18  | resnet18_train  |
| ResNet34  | resnet34_train  |
| ResNet50  | resnet50_train  |
| ResNet101  | resnet101_train  |
| ResNet152  | resnet152_train  |
| ResNeXt29_2x64d  | resnext292x64d_train  |
| ResNeXt29_4x64d  | ResNeXt29_4x64d  |
| ResNeXt29_8x64d  | ResNeXt29_8x64d  |
| ResNeXt29_32x4d  | ResNeXt29_32x4d  |
| SENet18  | SENet18  |
| ShuffleNetG2  | ShuffleNetG2  |
| ShuffleNetG3  | ShuffleNetG3  |
| ShuffleNetV2  | ShuffleNetV2  |
| SimpleDLA  | SimpleDLA  |
| UNetX  | UNetX  |
| VGG13  | VGG13  |
| VGG16  | VGG16  |
| VGG19  | VGG19  |

### Offline
```
 python3 main.py args.offline=True (or False)
 ```
