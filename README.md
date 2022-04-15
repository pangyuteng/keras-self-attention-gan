# keras-self-attention-gan


```
docker build -t attention .
docker run -it --runtime=nvidia -u $(id -u):$(id -g) -v /mnt:/mnt -w /workdir -v $PWD:/workdir attention bash


CUDA_VISIBLE_DEVICES=3 python $file


```

ref.
https://github.com/eriklindernoren/Keras-GAN
 
