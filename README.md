# WTCNet

## Requirements

- Pytorch version 0.4 or higher.
- Python version 3.0 or higher.

## Description

## Test on single image

We test a trained ResNet on a Trancos example image as follows:

```
python main.py -image_path figures/test.png \
                -model_path checkpoints/best_model_acacia_ResUnet.pth \
                -model_name ResUnet
```

or you can run the test.sh

```python
bash test.sh
#the content of an example is listed as bellow:
python main.py  -image_path ./figures/oilpalm/test_image.jpg \
                -model_path checkpoints/best_model_acacia_ResUnet.pth  \
                -model_name ResUnet
```

### Training the models from scratch

To train the model,

```
python main.py -m train -e oilpalm
```

for Acacia dataset

```python
python main.py -m train -e acacia
```

if you want to train other datasets by yourself, just change the -e parameter.





