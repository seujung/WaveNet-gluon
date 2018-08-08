# WaveNet with Gluon

Gluon implementation of [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)


## Requirements
- Python 3.6.1
- Mxnet 1.2
- tqdm
- scipy.io


## Usage

- arguments
  - batch_size : Define batch size (defualt=64)
  - epoches : Define total epoches (default=1000)
  - mu : Define mu (default=128)
  - n_residue : Define number of residue (default=24)
  - dilation_depth : Define dilation depth (default=10)
  - use_gpu : Use GPU  (default=True)
  - generation : generate wav file for model (default=True)

###### data generation
```
python sort-of-clevr.py
``` 

###### default setting
```
python main.py
``` 
or

###### manual setting
```
python main.py --batch_size=32 --epoches=100
```

## Results
![perf_acc](images/perf_result_acc.png)

![perf_loss](images/perf_result_l.png)

## Reference
- https://github.com/kimhc6028/relational-networks
