### keras-self-attention-gan

```
self attention gan module

can be used with 2d (or 3d, assuming you have a jumbo gpu) inputs.

ref.
https://arxiv.org/pdf/1805.08318.pdf
https://lilianweng.github.io/posts/2018-06-24-attention
https://github.com/eriklindernoren/Keras-GAN

```

### bare minimum... instructions

```

docker build -t attention .
docker run -it --runtime=nvidia --gpus=1 -u $(id -u):$(id -g) -w /workdir -v $PWD:/workdir attention bash

# original gan copied from https://github.com/eriklindernoren/Keras-GAN
cd gan
python gan.py

# self attention gan
cd sagan
python sagan.py


```
 
