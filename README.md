# Train CIFAR10 with PyTorch

This is a demo implementation of affine transformed sgd under randomly chozen generalized tangent kernel (GTK). It can be applied to any sgd-bsed optimizers, I simply use the current forked repo as an example.

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Training
```
# Start training with: 
python main.py --model mlp1 --act gelu --lr 0.1 --optimizer gtk --seed 0
```
