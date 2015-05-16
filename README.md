Reproducing MNIST results in [Ciresan 2012](http://arxiv.org/abs/1202.2745) with Theano
---

The code here builds on top of 3 supervised learning [theano tutorials](http://deeplearning.net/tutorial/) and another [library](https://github.com/rakeshvar/theanet) to implement the [best result](http://yann.lecun.com/exdb/mnist/) for classifying the famous mnist dataset at the time of writing.

The prediction process consists of:

- preprocessing
    - digit width normalization
    - elastic distortion of digits at train time
        - ![Multiple elastic distortions with sigma 8](./plots/distortions_8_sampled.png)
        - Multiple elastic distortions with sigma 8
- training DNN with SGD
    - 1x29x29-20C4-MP2-40C5-MP3-150N-10N
- testing with a committee of 35 Nets (5 nets per 7 choices of width normalization)

---

## Results:

Coming Soon...

## Hyper-parameters

### Sigma in elastic distortion

![Distortions varying with sigma, first row is original normalized image, second row is sigma=9, last is sigma=5](./plots/distortions_9_to_5.png)
Distortions varying with sigma, first row is original normalized image, second row is sigma=9, last is sigma=5

### Memory Consumption via Batch size

#### Number Parameters

This model has 171,940 parameters (`1*20*4*4 + 20 + 20*40*5*5 + 40 + 5*5*40*150 + 150 + 150*10 + 10`).

#### Number activations

This model has 4,741 activations (`1*29*29 + 13*13*20 + 3*3*40 + 150 + 10`).

#### Memory and Batch size

Each image per batch should take up 0.706 MB of GPU memory, so our batch size is not constrained in this case.
