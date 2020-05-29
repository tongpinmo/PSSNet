# WTCNet

## Requirements

- Pytorch version 0.4 or higher.
- Python version 3.0 or higher.

## Description



## Test on single image

We test a trained WTCNet on a acacia example image as follows:

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

## Experiments

#### 1: Download Datasets

* Acacia dataset & Oil Palm dataset

  ```python
  https://pan.baidu.com/s/1dawzKTGLt5Y_2PrDVNexdQ 
  Password: dnon
  ```

* Sorghum Plant 

  ```python
  https://engineering.purdue.edu/~sorghum/dataset-plant-centers-2016/
  ```

#### 2: Train the model

for Oil Palm dataset

```
python main.py -m train -e oilpalm
```

for Acacia dataset

```python
python main.py -m train -e acacia
```

If you want to train other datasets by yourself, just change the -e parameter.

### 3: Test the results

```python
python main.py -image_path figures/test.png \
                -model_path checkpoints/best_model_acacia_ResUnet.pth \
                -model_name ResUnet
```











