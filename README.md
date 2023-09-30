# vit
A PyTorch implementation of [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

<img src="assets/logo.png">

 
### Installation
```
git clone https://github.com/andregaio/vit.git
cd vit
conda create -n vit python=3.8
conda activate vit
pip install -r requirements.txt
```
### Models
| Name        |   Accuracy  |
| :---------- |   :------:  |
| ViTBase     |   ______    |
| ViTLarge    |   ______    |
| ViTHuge     |   ______    |

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