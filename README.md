# keras-self-attention-gan


```
docker build -t attention .
docker run -it --runtime=nvidia --gpus=1 -u $(id -u):$(id -g) -w /workdir -v $PWD:/workdir attention bash

# original gan
cd gan
python gan.py

# self attention gan
cd sagan
python sagan.py


```

ref.
https://github.com/eriklindernoren/Keras-GAN
 
