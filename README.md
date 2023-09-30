# vgg
A PyTorch implementation of [Very Deep Convolutional Networks For Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)

<img src="assets/logo.png">

 
### Installation
```
git clone https://github.com/andregaio/vgg.git
cd vgg
conda create -n vgg python=3.8
conda activate vgg
pip install -r requirements.txt
```
### Models
| Name       |   Accuracy  |
| :---------- |   :------:  |
| VGG-A       |   90.368    |
| VGG-A-LRN   |   90.813    |
| VGG-B       |   91.604    |
| VGG-C       |   91.822    |
| VGG-D       |   92.247    |
| VGG-E       |   91.901    |

### Dataset
- [CIFAR10](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html)

### Training
```
python train.py --model vgg_A --dataset cifar10
```

### Eval
```
python eval.py --model vgg_A --weights weights/checkpoint_00070.pt --dataset cifar10
```

### Inference
```
python infer.py --model vgg_A --weights weights/checkpoint_00070.pt --image assets/cat.png
```

### [Results](https://wandb.ai/andregaio/vgg)
<div align="center">


<img src="assets/chart.png">


</div>

### Notes
This implementation is not designed to be a complete replica of the original - the main differences are:
 - **Batchnorm** layers have been added prior to each activation
 - **Learning rate** modified to 10e-3
 - **Accuracy** has been used to evaluate classification performance
 - Has been trained on **CIFAR10**
 - Input resolution and FC layer sizes have been changed to **32x32** match dataset
 - **Automatic Mixed Precision (AMP)** training with gradient scaling and autocasting
 - **Kaiming** initialisation
 - **RGB colour shift** has not been used
 - **Dropout** set to 0.3 for all except vgg_C which was kept at 0.5
 - **Learning rate** dropped 2 times during training process