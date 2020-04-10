# MINet

## Folders & Files

* `base`: Store some basic code for building the network here.
* `loss`: The code of the loss function.
* `models`: The code of important modules.
* `network`: The code of the network.
* `utils`: Some instrumental code.
    * `utils/config.py`: Configuration file for model training and testing.
    * `utils/imgs/*py`: Some files about creating the dataloader.
* `main.py`: I think you can understand.

## My Environment

* Python=3.7
* PyTorch=1.3.1 (I think `>=1.1` is OK. You can try it, please create an issue if there is a problem :smile:.)
* tqdm
* prefetch-generator
* tensorboardX

Recommended way to install these packages:

```
# create env
conda create -n pt13 python=3.7
conda activate pt13

# install pytorch
conda install pytorch=1.3.1 torchvision cudatoolkit=9.0 cudnn -c pytorch

# some tools
pip install tqdm
# (optional) https://github.com/Lyken17/pytorch-OpCounter
# pip install thop
pip install prefetch-generator

# install tensorboard
pip install tensorboardX
pip install tensorflow==1.13.1
```

## Train

1. You can customize the value of the [`arg_config`](./utils/config.py#L20) dictionary in the configuration file.
    * The first time you use it, you need to adjust the [path](./utils/config.py#L9-L17) of every dataset.
2. In the folder `code`, run the command `python main.py`.
3. Everything is OK. Just wait for the results.
4. The test will be performed automatically when the training is completed.
5. All results will be saved into the folder `output`, including predictions in folder `pre` (if you set `save_pre` to `True`), `.pth` files in folder `pth` and other log files.

## If you want to test the trained model again...

**Our pre-training parameters can also be used in this way.**

1. In the `output` folder, please ensure that there is a folder corresponding to the model, which contains the `pth` folder, and the `.pth` file of the model is located here.
2. Set the value of `NET` of `arg_config` to the model you want to test like these [lines](utils/config.py#L27-L30).
3. Set the value of `epoch_num` to 0.
4. In the folder `code`, run `python main.py`.
5. You can find predictions from the model in the folder `pre` of the `output`.

## Evaluation

We evaluation results of all models by [ArcherFMY/sal_eval_toolbox](https://github.com/ArcherFMY/sal_eval_toolbox/tree/master/tools). And we add the code about E-measure and weighted F-measure and update the related code in our forked repository [lartpang/sal_eval_toolbox](https://github.com/lartpang/SODEvalToolkit/tree/master/tools). Welcome to use it in your code :star:!

## More

If there are other issues, you can create a new issue.
