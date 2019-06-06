# Mean Teacher Demo on CIFAR10

This is a reimplementation of Mean Teacher by Vaipola et al
using CIFAR10 and WResNet as the backend, as suggested in a recent
Google Brain paper.

## Running

To run it, use something like

```
python train.py \
    --batch-size=128 \
    --supervised-ratio=0.1 \
    --noise=0.0 \
    --regularizer=mt \
    --epochs 300 \
    --consistency-weight 1 \
    --learning-rate=0.02 \
    --cuda \
    --save-to model.pt
```

To load an existing model and test on the validation set:

```
python train.py \
    --batch-size=128 \
    --supervised-ratio=0.1 \
    --noise=0.0 \
    --regularizer=mt \
    --epochs 300 \
    --consistency-weight 1 \
    --learning-rate=0.02 \
    --cuda \
    --load model.pt \
    --test-only
```

There is no need to download any data, we use the built-in
CIFAR-10 dataset in torchvision which will be downloaded automatically.

The model will be saved to `model.pt` and train/validation loss/accuracy
curves asaved to `model.pt.log`. Use `--supervised-ratio` to specify
what percentage of each class should be labelled, the rest will have
a label of -1.

Use `--regularizer=mt` to use Mean Teacher as the learned regularizer.
Use `--reguarlizer=none` to use no learned reguarlizer (only dropout
and weight decay).

## Results

Validation set results on CIFAR10, with 10% of data labelled:

| Model        | Loss     | Accuracy    |
|--------------|----------|-------------|
| WResNet MT   | 0.6074   | 0.876       |
| WResNet None | 0.8448   | 0.813       |
| ResNet  MT   | 0.3938   | 0.899       |
| ResNet None  | 0.8438   | 0.826       |

