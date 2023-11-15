# Efficient Unsupervised Video Object Segmentation Network Based on Motion
Guidance

## Test Environment
- Ubuntu 
- python 3.6
- Pytorch 0.3.1
  + installed with CUDA.



## How to Run
1) Download [DAVIS-2017](https://davischallenge.org/davis2017/code.html).
2) Edit path for `DAVIS_ROOT` in run.py.
``` python
DAVIS_ROOT = '<Your DAVIS path>'
```
3) Download [weights.pth](https://www.dropbox.com/s/gt0kivrb2hlavi2/weights.pth?dl=0) and place it the same folde as run.py.
4) To run single-object video object segmentation on DAVIS-2016 validation.
``` 
python run.py
```
5) To run multi-object video object segmentation on DAVIS-2017 validation.
``` 
python run.py -MO
```
6) Results will be saved in `./results/SO` or `./results/MO`.





  










