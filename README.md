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
| DenseNet161  | Content Cell  |
| DenseNet169  | Content Cell  |
| DenseNet201  | Content Cell  |
| densenet_cifar  | Content Cell  |
| DLA  | Content Cell  |
| DPN26  | Content Cell  |
| DPN92  | Content Cell  |
| EfficientNetB0  | Content Cell  |
| GoogLeNet  | Content Cell  |
| MobileNet  | Content Cell  |
| MobileNetV2  | Content Cell  |
| PNASNetA  | Content Cell  |
| PNASNetB  | Content Cell  |
| PreActResNet18  | Content Cell  |
| PreActResNet34  | Content Cell  |
| PreActResNet50  | Content Cell  |
| PreActResNet101  | Content Cell  |
| PreActResNet152  | Content Cell  |
| RegNetX_200MF  | Content Cell  |
| RegNetX_400MF  | Content Cell  |
| RegNetY_400MF  | Content Cell  |
| ResNet18  | Content Cell  |
| ResNet34  | Content Cell  |
| ResNet50  | Content Cell  |
| ResNet101  | Content Cell  |
| ResNet152  | Content Cell  |
| ResNeXt29_2x64d  | Content Cell  |
| ResNeXt29_4x64d  | Content Cell  |
| ResNeXt29_8x64d  | Content Cell  |
| ResNeXt29_32x4d  | Content Cell  |
| SENet18  | Content Cell  |
| ShuffleNetG2  | Content Cell  |
| ShuffleNetG3  | Content Cell  |
| ShuffleNetV2  | Content Cell  |
| SimpleDLA  | Content Cell  |
| UNetX  | Content Cell  |
| VGG13  | Content Cell  |
| VGG16  | Content Cell  |
| VGG19  | Content Cell  |
