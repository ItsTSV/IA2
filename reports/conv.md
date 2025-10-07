# Own convolutional neural network
Architecture stays the same in all experiments (because using Optuna would burn my already charred notebook). 
Optimizer is always AdamW. The number of epochs was always 100, but it was by no means necessary.
The only thing that changes is kernel sizes, strides, dropouts and learning rate.

### Table
| Kernel Sizes | Strides   | LR      | Dropouts    | Accuracy |
|--------------|-----------|---------|-------------|----------|
| (3, 3, 3)    | (1, 1, 1) | 0.001   | (0.4, 0.2)  | 0.642    |
| (5, 3, 2)    | (3, 2, 1) | 0.001   | (0.3, 0.3)  | 0.656    |
| (5, 3, 2)    | (3, 2, 2) | 0.00001 | (0.3, 0.2)  | 0.737    |


# Transfer Learning: MobileNetV3 Large

