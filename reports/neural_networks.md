# Augmentations
- Grayscale
- Jitter (brightness, contrast, saturation = 0.4; hue = 0.2)
- Resize (40, 60) for my models; (244, 244) for transfer learning
- Tensor

# Own fully connected neural network
Very basic architecture:

```
Linear -> BatchNorm -> ReLU -> Linear -> BatchNorm -> ReLU -> Linear
```

Used optimizer is SGD with momentum 0.9 and lr 0.001. Also tried Adagrad and AdamW, but they
were performing a bit worse. The number of training epochs was always 100, which was by no means necessary.

**Accuracy: 0.865**

# Own convolutional neural network
Architecture stays the same in all experiments (because using Optuna would burn my already charred notebook):

Optimizer is always AdamW; other ones were performing worse. The number of epochs was always 100, but it was by no means necessary.
The only thing that changes is kernel sizes, strides, dropouts and learning rate.

### Changed hyperparameters
| Kernel Sizes | Strides   | LR      | Dropouts    | Accuracy |
|--------------|-----------|---------|-------------|----------|
| (3, 3, 3)    | (1, 1, 1) | 0.001   | (0.4, 0.2)  | 0.642    |
| (5, 3, 2)    | (3, 2, 1) | 0.001   | (0.3, 0.3)  | 0.656    |
| (5, 3, 2)    | (3, 2, 2) | 0.00001 | (0.3, 0.2)  | 0.737    |

**Accuracy: 0.922**

# Transfer Learning: MobileNetV3 Large

