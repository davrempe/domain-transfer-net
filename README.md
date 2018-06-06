# Before Getting Started

* **Face transfer showed very limited success compared to the original paper.** Please see our [final report](http://cs229.stanford.edu/proj2017/final-reports/5241608.pdf) for a discussion of achievable results with this implementation.
* To generate the dataset of emoji images:
   * Create a directory called `emoji_data` in the current `datasets` directory
   * There are 2 emoji generation scripts: [`create_emojis.py`](https://github.com/davrempe/domain-transfer-net/blob/master/datasets/create_emojis.py) is for creating small datasets (<1000 images) and [`create_emojis_parallel.py`](https://github.com/davrempe/domain-transfer-net/blob/master/datasets/create_emojis_parallel.py) for large datasets.
   * Run `python3 create_emojis.py` or `python3 create_emojis_parallel.py` to generate the dataset. Note **you must use python3** to run these scripts because of the url request. 
   * To change the number of emojis generated, change the `num_emojis` variable at the top of the generation script.

# CS 229 Final Project: Unsupervised Cross-Domain Image Generation

## Davis Rempe, Xinru Hua, Haotian Zhang (Team 925)

Based on [this paper](https://arxiv.org/abs/1611.02200).

### Overview of files in this repo:
* `/datasets/` - all dataset download/creation/processing scripts
* `/pretrained_model/` - all saved pretrained models for f blocks
* Network for digit domain transfer:
    * `digits_model.py`
* Networks for face domain transfer:
    * `faces_model.py` - network as in paper
    * `faces_model_v2.py` - modified network for testing
    * `net_sphere.py` - SphereFace network (taken mostly from [here](https://github.com/clcarwin/sphereface_pytorch))
    * `open_face_model.py` and `SpatialCrossMapLRN_model.py` - OpenFace network (taken mostly from [here](https://github.com/thnkim/OpenFacePytorch))
* Training scripts:
    * `base_test.py` - abstract training class
    * `classifier_f_test.py` - f block training for digit transfer
    * `digit_model_test.py` - digit model transfer training
    * `digit_model_test_septrain.py` - digit model transfer training with separated source and target training
    * `faces_model_test_open.py` - face transfer training with OpenFace
    * `faces_model_test_sphere.py` - face transfer training with SphereFace
* Training script drivers:
    * `FaceMain.ipynb` - driver for training face transfer
* `data.py` - all data loaders and preprocessing code

### Other references
* digit model architecture inspired by [this implementation](https://github.com/taey16/DomainTransferNetwork.pytorch)
* training strategies inspired by [this implementation](https://github.com/yunjey/domain-transfer-network)
