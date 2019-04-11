# Discovering hidden factors of variation in deep networks
[![MIT license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](https://github.com/Warvito/discovering-hidden-factors-of-variation-in-deep-networks/blob/master/LICENSE)

 This is a Tensorflow 2.0 implementation of [Discovering hidden factors of variation in deep networks](https://arxiv.org/abs/1412.6583) by [Brian Cheung](https://twitter.com/thisismyhat) et al. (2014). This repository contains reproduce of several experiments mentioned in the paper. Based on the Lasagne (RIP) example ([link](https://github.com/Lasagne/Lasagne/blob/highway_example/examples/Hidden%20factors.ipynb)).
 
## Abstract
Deep learning has enjoyed a great deal of success because of its ability to learnuseful  features  for  tasks  such  as  classification.   But  there  has  been  less  explo-ration in learning the factors of variation apart from the classification signal.  Byaugmenting  autoencoders  with  simple  regularization  terms  during  training,  wedemonstrate  that  standard  deep  architectures  can  discover  and  explicitly  repre-sent factors of variation beyond those relevant for categorization.  We introducea cross-covariance penalty (XCov) as a method to disentangle factors like hand-writing style for digits and subject identity in faces.  We demonstrate this on theMNIST handwritten digit database,  the Toronto Faces Database (TFD) and theMulti-PIE dataset by generating manipulated instances of the data.  Furthermore,we demonstrate these deep networks can extrapolate ‘hidden’ variation in the supervised signal.
 
## Requirements
- Python 3
- [TensorFlow 2.0+](https://www.tensorflow.org/)
- [Numpy](http://www.numpy.org/)
- [Matplotlib](https://matplotlib.org/)


## Installing the dependencies
Install virtualenv and creating a new virtual environment:

    pip install virtualenv
    virtualenv -p /usr/bin/python3 ./venv

Install dependencies

    pip3 install -r requirements.txt



## Disclaimer
This is not an official implementation.


## Citation
If you find this code useful for your research, please cite:

    @article{cheung2014discovering,
      title={Discovering hidden factors of variation in deep networks},
      author={Cheung, Brian and Livezey, Jesse A and Bansal, Arjun K and Olshausen, Bruno A},
      journal={arXiv preprint arXiv:1412.6583},
      year={2014}
    }