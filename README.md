# PriorNet


## dependencies 
```
>= Ubuntu 16.04 
>= Python 3.7
>= Pytorch 1.3.0
OpenCV-Python
```

## preparation 
- download the official pretrained model ([Baidu drive](https://pan.baidu.com/s/1zRhAaGlunIZEOopNSxZNxw 
code：fv6m)) of ResNet-50 implemented in Pytorch if you want to train the network again.
- download or put the RGB saliency benchmark datasets ([Baidu drive](https://pan.baidu.com/s/1kUPZGSe1CN4AOVmB3R3Qxg 
code：sfx6)) in the folder of `data` for training or test.

## training
you may revise the `TAG` and `SAVEPATH` defined in the *train.py*. After the preparation, run this command 
```
'CUDA_VISIBLE_DEVICES=0,1,…… python -m torch.distributed.launch --nproc_per_node=x train.py -b 16'
```
make sure that the GPU memory is enough (You can adjust the batch according to your GPU memory).

## test
After the preparation, run this commond to generate the final saliency maps.
```
 python test.py 
```

We provide the trained model file ([Baidu drive](link：https://pan.baidu.com/s/1Io_rOuojuCVnjteH0N1Pdw?pwd=q6z5 code：q6z5), and run this command to check its completeness:
```
cksum PriorNet 
```
you will obtain the result `PriorNet`.

## evaluation

We provide the predicted saliency maps on five benchmark datasets,including PASCAL-S, ECSSD, HKU-S, DUT-OMRON and DUTS-TE. ([Baidu drive](link https://pan.baidu.com/s/1sgSfukdvjg-kEF7fnhM74Q?pwd=29m7 code：29m7)

You can use the evaluation code in the folder  "eval_code" for fair comparisons, but you may need to revise the `algorithms` , `data_root`, and `maps_root` defined in the `main.m`. 
