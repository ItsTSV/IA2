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

# Transfer Learning: MobileNetV3 Small
I picked small MobileNetV3 because it was by far the fastest model to train. The 
parameters : accuracy ratio is very good (it only has 2.5M params, approximately 9.8MB). 

I performed few changes to the architecture:
- One channel input (instead of three)
- Changed classifier to my own (two linear layers, dropout, one neuron output)
- Froze the whole model except the classifier

For training, I used AdamW with lr 0.0001. The number of epochs was 10, because
the model is already pretrained (and also pretty big for training on CPU). After that, I fined tuned it again, with lr 0.00005 for 5 epochs.
And it just worked.

**Accuracy: 0.975**

# Transfer Learning: EfficientNetB0
In order to protect nature and our household's electricity bills, I picked another very lite model: EfficientNetB0.
It has 5.3M params (approximately 33MB), which is still pretty okay. The changes to the architecture were the same
as for the MobileNetV3. The training was also the same -- first 10 epochs with lr 0.0001, then some more epochs with lower
learning rate to catch some weird edge cases (nighttime + parking spot under the lamp, sunny + parking spot in the shadow of a tree, etc...)
And again, it just worked -- as well as MobileNetV3 ;)

**Accuracy: 0.976**

# Fancy chart to sum it up
![Model Comparison](model_comparison.png)

