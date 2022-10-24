# LTE-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation
of [Local Texture Estimator for Implicit Representation Function](https://arxiv.org/pdf/2111.08918v6.pdf).

## Table of contents

- [LTE-PyTorch](#lte-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [How Test and Train](#how-test-and-train)
        - [Test](#test)
        - [Train LTE model](#train-lte-model)
        - [Resume train LTE model](#resume-train-lte-model)
    - [Result](#result)
    - [Contributing](#contributing)
    - [Credit](#credit)
        - [Local Texture Estimator for Implicit Representation Function](#local-texture-estimator-for-implicit-representation-function)

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

Contains DIV2K, DIV8K, Flickr2K, OST, T91, Set5, Set14, BSDS100 and BSDS200, etc.

- [Google Driver](https://drive.google.com/drive/folders/1A6lzGeQrFMxPqJehK9s37ce-tPDj20mD?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1o-8Ty_7q6DiS3ykLU09IVg?pwd=llot)

Please refer to `README.md` in the `data` directory for the method of making a dataset.

## How Test and Train

Both training and testing only need to modify the `config.py` file.

### Test

modify the `config.py`

- line 31: `model_arch_name` change to `lte_edsr`.
- line 37: `upscale_factor` change to `4`.
- line 39: `mode` change to `test`.
- line 89: `model_weights_path` change to `./results/pretrained_models/LTE_EDSR_x4-DIV2K-91316a4b.pth.tar`.
-

```bash
python3 test.py
```

### Train LTE model

modify the `config.py`

- line 31: `model_arch_name` change to `lte_edsr`.
- line 37: `upscale_factor` change to `4`.
- line 39: `mode` change to `train`.

```bash
python3 train.py
```

### Resume train LTE model

modify the `lte.py`

- line 31: `model_arch_name` change to `lte_edsr`.
- line 37: `upscale_factor` change to `4`.
- line 39: `mode` change to `train`.
- line 58: `resume_model_weights_path` change to `./results/LTE_EDSR-DIV2K/epoch_xxx.pth.tar`.

```bash
python3 train.py
```

## Result

Source of original paper results: [https://arxiv.org/pdf/2111.08918v6.pdf](https://arxiv.org/pdf/2111.08918v6.pdf)

In the following table, the psnr value in `()` indicates the result of the project, and `-` indicates no test.

|  Method  | Scale |      Set5 (PSNR/SSIM)      |     Set14 (PSNR/SSIM)      | 
|:--------:|:-----:|:--------------------------:|:--------------------------:|
| LTE_EDSR |   2   |     -(**-**)/-(**-**)      |     -(**-**)/-(**-**)      | 
| LTE_EDSR |   3   |     -(**-**)/-(**-**)      |     -(**-**)/-(**-**)      |
| LTE_EDSR |   4   | -(**32.12**)/-(**0.8653**) | -(**28.50**)/-(**0.7811**) | 


```bash
# Download `LTE_EDSR_x4-DIV2K-91316a4b.pth.tar.pth.tar` weights to `./results/pretrained_models`
# More detail see `README.md<Download weights>`
python3 ./inference.py
```

Input:

<span align="center"><img width="492" height="480" src="figure/baboon_lr.png"/></span>

Output:

<span align="center"><img width="492" height="480" src="figure/baboon_sr.png"/></span>

```text
Build `lte_edsr` model successfully.
Load `lte_edsr` model weights `./results/pretrained_models/LTE_EDSR_x4-DIV2K-91316a4b.pth.tar` successfully.
SR image save to `./figure/baboon_lr.png`
```

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

## Credit

### Local Texture Estimator for Implicit Representation Function

_Jaewon Lee, Kyong Hwan Jin_ <br>

**Abstract** <br>
Recent works with an implicit neural function shed light on representing images in arbitrary resolution. However, a
standalone multi-layer perceptron shows limited performance in learning high-frequency components. In this paper, we
propose a Local Texture Estimator (LTE), a dominant-frequency estimator for natural images, enabling an implicit
function to capture fine details while reconstructing images in a continuous manner. When jointly trained with a deep
super-resolution (SR) architecture, LTE is capable of characterizing image textures in 2D Fourier space. We show that an
LTE-based neural function achieves favorable performance against existing deep SR methods within an arbitrary-scale
factor. Furthermore, we demonstrate that our implementation takes the shortest running time compared to previous works.

[[Paper]](https://arxiv.org/pdf/2111.08918v6.pdf) [[Code]](https://github.com/jaewon-lee-b/lte)

```bibtex
@InProceedings{lte-jaewon-lee,
    author    = {Lee, Jaewon and Jin, Kyong Hwan},
    title     = {Local Texture Estimator for Implicit Representation Function},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {1929-1938}
}
```
