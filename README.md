# domain-transfer-net

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
    * `digit_model_test_sep_Haotian.py` - training using some of Haotians methods
    * `faces_model_test_open.py` - face transfer training with OpenFace
    * `faces_model_test_sphere.py` - face transfer training with SphereFace
* Training script drivers:
    * `FaceMain.ipynb` - driver for training face transfer
* `data.py` - all data loaders and preprocessing code
