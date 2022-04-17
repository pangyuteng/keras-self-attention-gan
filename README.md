# keras-self-attention-gan


```
docker build -t attention .
docker run -it --runtime=nvidia --gpus=1 -u $(id -u):$(id -g) -w /workdir -v $PWD:/workdir attention bash

# original gan and bigan
# copied from https://github.com/eriklindernoren/Keras-GAN

cd gan
python gan.py
cd bigan
python bigan.py


# self attention gan
cd sagan
python sagan.py


# WIP 3d self attention gan
cd 3dsagan
python 3dsagan.py

```

ref.
https://github.com/eriklindernoren/Keras-GAN
 
