The implementation of the [SCPNet](https://arxiv.org/abs/1810.06996) modified from [person-reid-triplet-loss-baseline](https://github.com/huanghoujing/person-reid-triplet-loss-baseline). If you use this repo, please cite the following paper:

```latex
@inproceedings{SCPNet,
  title = {SCPNet: Spatial-Channel Parallelism Network for Joint Holistic and Partial Person Re-Identification},
  author = {Fan, Xing and Luo, Hao and Zhang, Xuan and He, Lingxiao and Zhang, Chi and Jiang, Wei},
  booktitle = {ACCV},
  year = {2018}
}
```

# Installation

The original code is based on a internal deep learning framewok using 4 datasets together. We re-implement it using PyTorch in this repo based on  [person-reid-triplet-loss-baseline](https://github.com/huanghoujing/person-reid-triplet-loss-baseline) (More useful information may be found).

You can use Python2 and install PyTorch using the following commands (at least one Nvida GPU is required):

```sh
pip install torch==0.3.1
pip install torchvision
```

After training on single dataset alone for 200 epoch, the following results should be achieved:

|                          | Market-1501             | DukeMTMC-reID           |
| ------------------------ | ----------------------- | ----------------------- |
| Original Version (paper) | rank-1: 91.2, mAP: 75.2 | rank-1: 80.3, mAP: 62.6 |
| This e-implement (repo)  | rank-1: 90.4, mAP: 74.9 | rank-1: 81.2, mAP: 64.5 |

# Dataset Preparation

Inspired by Tong Xiao's [open-reid](https://github.com/Cysu/open-reid) project, you need to prepare datasets first.

## Market1501

You can download what I have transformed for the project from [Google Drive](https://drive.google.com/open?id=1CaWH7_csm9aDyTVgjs7_3dlZIWqoBlv4) or [BaiduYun](https://pan.baidu.com/s/1nvOhpot). Otherwise, you can download the original dataset and transform it using my script, described below.

Download the Market1501 dataset from [here](http://www.liangzheng.org/Project/project_reid.html). Run the following script to transform the dataset, replacing the paths with yours.

```bash
python script/dataset/transform_market1501.py \
--zip_file ~/Dataset/market1501/Market-1501-v15.09.15.zip \
--save_dir ~/Dataset/market1501
```

## CUHK03

We follow the new training/testing protocol proposed in paper
```
@article{zhong2017re,
  title={Re-ranking Person Re-identification with k-reciprocal Encoding},
  author={Zhong, Zhun and Zheng, Liang and Cao, Donglin and Li, Shaozi},
  booktitle={CVPR},
  year={2017}
}
```
Details of the new protocol can be found [here](https://github.com/zhunzhong07/person-re-ranking).

You can download what I have transformed for the project from [Google Drive](https://drive.google.com/open?id=1Ssp9r4g8UbGveX-9JvHmjpcesvw90xIF) or [BaiduYun](https://pan.baidu.com/s/1hsB0pIc). Otherwise, you can download the original dataset and transform it using my script, described below.

Download the CUHK03 dataset from [here](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html). Then download the training/testing partition file from [Google Drive](https://drive.google.com/open?id=14lEiUlQDdsoroo8XJvQ3nLZDIDeEizlP) or [BaiduYun](https://pan.baidu.com/s/1miuxl3q). This partition file specifies which images are in training, query or gallery set. Finally run the following script to transform the dataset, replacing the paths with yours.

```bash
python script/dataset/transform_cuhk03.py \
--zip_file ~/Dataset/cuhk03/cuhk03_release.zip \
--train_test_partition_file ~/Dataset/cuhk03/re_ranking_train_test_split.pkl \
--save_dir ~/Dataset/cuhk03
```


## DukeMTMC-reID

You can download what I have transformed for the project from [Google Drive](https://drive.google.com/open?id=1P9Jr0en0HBu_cZ7txrb2ZA_dI36wzXbS) or [BaiduYun](https://pan.baidu.com/s/1miIdEek). Otherwise, you can download the original dataset and transform it using my script, described below.

Download the DukeMTMC-reID dataset from [here](https://github.com/layumi/DukeMTMC-reID_evaluation). Run the following script to transform the dataset, replacing the paths with yours.

```bash
python script/dataset/transform_duke.py \
--zip_file ~/Dataset/duke/DukeMTMC-reID.zip \
--save_dir ~/Dataset/duke
```


## Combining Trainval Set of Market1501, CUHK03, DukeMTMC-reID

Larger training set tends to benefit deep learning models, so I combine trainval set of three datasets Market1501, CUHK03 and DukeMTMC-reID. After training on the combined trainval set, the model can be tested on three test sets as usual.

Transform three separate datasets as introduced above if you have not done it.

For the trainval set, you can download what I have transformed from [Google Drive](https://drive.google.com/open?id=1hmZIRkaLvLb_lA1CcC4uGxmA4ppxPinj) or [BaiduYun](https://pan.baidu.com/s/1jIvNYPg). Otherwise, you can run the following script to combine the trainval sets, replacing the paths with yours.

```bash
python script/dataset/combine_trainval_sets.py \
--market1501_im_dir ~/Dataset/market1501/images \
--market1501_partition_file ~/Dataset/market1501/partitions.pkl \
--cuhk03_im_dir ~/Dataset/cuhk03/detected/images \
--cuhk03_partition_file ~/Dataset/cuhk03/detected/partitions.pkl \
--duke_im_dir ~/Dataset/duke/images \
--duke_partition_file ~/Dataset/duke/partitions.pkl \
--save_dir ~/Dataset/market1501_cuhk03_duke
```

## Configure Dataset Path

The project requires you to configure the dataset paths. In `tri_loss/dataset/__init__.py`, modify the following snippet according to your saving paths used in preparing datasets.

```python
# In file tri_loss/dataset/__init__.py

########################################
# Specify Directory and Partition File #
########################################

if name == 'market1501':
  im_dir = ospeu('~/Dataset/market1501/images')
  partition_file = ospeu('~/Dataset/market1501/partitions.pkl')

elif name == 'cuhk03':
  im_type = ['detected', 'labeled'][0]
  im_dir = ospeu(ospj('~/Dataset/cuhk03', im_type, 'images'))
  partition_file = ospeu(ospj('~/Dataset/cuhk03', im_type, 'partitions.pkl'))

elif name == 'duke':
  im_dir = ospeu('~/Dataset/duke/images')
  partition_file = ospeu('~/Dataset/duke/partitions.pkl')

elif name == 'combined':
  assert part in ['trainval'], \
    "Only trainval part of the combined dataset is available now."
  im_dir = ospeu('~/Dataset/market1501_cuhk03_duke/trainval_images')
  partition_file = ospeu('~/Dataset/market1501_cuhk03_duke/partitions.pkl')
```

## Evaluation Protocol

Datasets used in this project all follow the standard evaluation protocol of Market1501, using CMC and mAP metric. According to [open-reid](https://github.com/Cysu/open-reid), the setting of CMC is as follows

```python
# In file tri_loss/dataset/__init__.py

cmc_kwargs = dict(separate_camera_set=False,
                  single_gallery_shot=False,
                  first_match_break=True)
```

To play with [different CMC options](https://cysu.github.io/open-reid/notes/evaluation_metrics.html), you can [modify it accordingly](https://github.com/Cysu/open-reid/blob/3293ca79a07ebee7f995ce647aafa7df755207b8/reid/evaluators.py#L85-L95).

```python
# In open-reid's reid/evaluators.py

# Compute all kinds of CMC scores
cmc_configs = {
  'allshots': dict(separate_camera_set=False,
                   single_gallery_shot=False,
                   first_match_break=False),
  'cuhk03': dict(separate_camera_set=True,
                 single_gallery_shot=True,
                 first_match_break=False),
  'market1501': dict(separate_camera_set=False,
                     single_gallery_shot=False,
                     first_match_break=True)}
```


# Examples


## Test

My training log and saved model weights for three datasets can be downloaded from [Google Drive](https://drive.google.com/open?id=14ljnClpZkHD7BzrET1q1eFQ_XhaRzM3-) or [BaiduYun](https://pan.baidu.com/s/1mjfTcxy).

Specify
- a dataset name (one of `market1501`, `cuhk03`, `duke`)
- stride, `1` or `2`
- an experiment directory for saving testing log
- the path of the downloaded `model_weight.pth`

in the following command and run it.

```bash
python2 script/experiment/train.py \
-d '(0,)' \
--only_test true \
--dataset DATASET_NAME \
--last_conv_stride STRIDE \
--normalize_feature false \
--exp_dir EXPERIMENT_DIRECTORY \
--model_weight_file THE_DOWNLOADED_MODEL_WEIGHT_FILE
```

## Train

You can also train it by yourself. The following command performs training, validation and finally testing automatically.

Specify
- a dataset name (one of `['market1501', 'cuhk03', 'duke']`)
- stride, `1` or `2`
- training on `trainval` set or `train` set (for tuning parameters)
- an experiment directory for saving training log

in the following command and run it.

```bash
python2 script/experiment/train.py \
-d '(0,)' \
--only_test false \
--dataset DATASET_NAME \
--last_conv_stride STRIDE \
--normalize_feature false \
--trainset_part TRAINVAL_OR_TRAIN \
--exp_dir EXPERIMENT_DIRECTORY \
--steps_per_log 10 \
--epochs_per_val 5
```

### Log

During training, you can run the [TensorBoard](https://github.com/lanpa/tensorboard-pytorch) and access port `6006` to watch the loss curves etc. E.g.

```bash
# Modify the path for `--logdir` accordingly.
tensorboard --logdir YOUR_EXPERIMENT_DIRECTORY/tensorboard
```

For more usage of TensorBoard, see the website and the help:

```bash
tensorboard --help
```


## Visualize Ranking List

Specify
- a dataset name (one of `['market1501', 'cuhk03', 'duke']`)
- stride, `1` or `2`
- either `model_weight_file` (the downloaded `model_weight.pth`) OR `ckpt_file` (saved `ckpt.pth` during training)
- an experiment directory for saving images and log

in the following command and run it.

```bash
python script/experiment/visualize_rank_list.py \
-d '(0,)' \
--num_queries 16 \
--rank_list_size 10 \
--dataset DATASET_NAME \
--last_conv_stride STRIDE \
--normalize_feature false \
--exp_dir EXPERIMENT_DIRECTORY \
--model_weight_file '' \
--ckpt_file ''
```

Each query image and its ranking list would be saved to an image in directory `EXPERIMENT_DIRECTORY/rank_lists`. As shown in following examples, green boundary is added to true positive, and red to false positve.

![](example_rank_lists_on_Duke/00000126_0002_00000021.jpg)

![](example_rank_lists_on_Duke/00000147_0003_00000004.jpg)

![](example_rank_lists_on_Duke/00000169_0001_00000008.jpg)

![](example_rank_lists_on_Duke/00000257_0003_00000004.jpg)

# References & Credits

- [SCPNet](https://arxiv.org/abs/1810.06996)
- [person-reid-triplet-loss-baseline](https://github.com/huanghoujing/person-reid-triplet-loss-baseline)

- [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737)
- [Beyond Part Models: Person Retrieval with Refined Part Pooling](https://arxiv.org/abs/1711.09349)
- [open-reid](https://github.com/Cysu/open-reid)
- [Re-ranking Person Re-identification with k-reciprocal Encoding](https://github.com/zhunzhong07/person-re-ranking)
- [Market1501](http://www.liangzheng.org/Project/project_reid.html)
- [CUHK03](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html)
- [DukeMTMC-reID](https://github.com/layumi/DukeMTMC-reID_evaluation)
