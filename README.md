# CORRDROP

This is our Pytorch implementation of CorrDrop.

**Corrdrop: Correlation Based Dropout for Convolutional Neural Networks (ICASSP 2020)**

**Correlation-based structural dropout for convolutional neural networks (Pattern Recognition)**

## Setting

Set your data directory in ./train/dataLoader.py 

```python
data_dir = './data'
```

## Train model with SCD

train without cutout

```
python3 train.py --dropway 'SCD' --dataset 'cifar-10' --depth 20 --gpu_ids 0 --batch_size 128 --epoch 200 --cutout 0 --exp_dir './exp/scd' --model 'resnet' --p 0.2 --blocksize 5
```

train with cutout

```python
python3 train.py --dropway 'SCD' --dataset 'cifar-10' --depth 20 --gpu_ids 0 --batch_size 128 --epoch 200 --cutout 1 --exp_dir './exp/scd' --model 'resnet' --p 0.03 --blocksize 5
```

## Train model with CCD

train without cutout

```python
python3 train.py --dropway 'CCD' --dataset 'cifar-100' --depth 20 --gpu_ids 0 --batch_size 128 --epoch 200 --cutout 0 --exp_dir './exp/ccd' --model 'resnet' --p 0.2
```

train with cutout

```python
python3 train.py --dropway 'CCD' --dataset 'cifar-100' --depth 20 --gpu_ids 0 --batch_size 128 --epoch 200 --cutout 1 --exp_dir './exp/ccd' --model 'resnet' --p 0.1
```

## Citations

```
@inproceedings{zeng2020corrdrop,
  title={Corrdrop: Correlation based dropout for convolutional neural networks},
  author={Zeng, Yuyuan and Dai, Tao and Xia, Shu-Tao},
  booktitle={ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={3742--3746},
  year={2020},
  organization={IEEE}
}
```

```
@article{zeng2021correlation,
  title={Correlation-based structural dropout for convolutional neural networks},
  author={Zeng, Yuyuan and Dai, Tao and Chen, Bin and Xia, Shu-Tao and Lu, Jian},
  journal={Pattern Recognition},
  volume={120},
  pages={108117},
  year={2021},
  publisher={Elsevier}
}
```

