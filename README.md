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
