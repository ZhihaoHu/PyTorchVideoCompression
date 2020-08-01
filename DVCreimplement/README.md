# DVC: An End-to-end Deep Video Compression Framework

PyTorch reimplemetation for the paper:

DVC: An End-to-end Deep Video Compression Framework, Guo Lu, Wanli Ouyang, Dong Xu, Xiaoyun Zhang, Chunlei Cai, Zhiyong Gao, CVPR 2019 (**Oral**). [[arXiv]](https://arxiv.org/abs/1812.00101)


## Requirements

- Python==3.6
- PyTorch==1.2

## Data Preparation

### Training data

1. Download [Vimeo-90k dataset](http://toflow.csail.mit.edu/): original training + test set (82GB)

2. Unzip the dataset into `./data/`.

### Test data

This method only provide P-frame compression, so we first need to generate I frames by H.265. We take UVG dataset as an example.

1. Download [UVG dataset](http://ultravideo.cs.tut.fi/#testsequences_x)(1080p/8bit/YUV/RAW) to `data/UVG/videos/`.
2. Crop Videos from 1920x1080 to 1920x1024.
    ```
    cd data/UVG/
    ffmpeg -pix_fmt yuv420p  -s 1920x1080 -i ./videos/xxxx.yuv -vf crop=1920:1024:0:0 ./videos_crop/xxxx.yuv
    ```
3. Convert YUV files to images.
    ```
    python convert.py
    ```
4. Create I frames. We need to create I frames by H.265 with $crf of 20,23,26,29.
    ```
    cd CreateI
    sh h265.sh $crf 1920 1024
    ```
    After finished the generating of I frames of each crf, you need to use bpps of each video in `result.txt` to fill the bpps in Class UVGdataset in `dataset.py`.

## Training
    cd examples/example
    sh cp.sh
    sh run.sh
If you want models with more λ, you can edit `config.json`

If you want to use tensorboard:

    cd examples
    sh tf.sh xxxx

## Testing
    sh test.sh

## Citation
If you find this paper useful, please cite:
```
@article{lu2018dvc,
  title={DVC: An End-to-end Deep Video Compression Framework},
  author={Lu, Guo and Ouyang, Wanli and Xu, Dong and Zhang, Xiaoyun and Cai, Chunlei and Gao, Zhiyong},
  journal={arXiv preprint arXiv:1812.00101},
  year={2018}
}
```