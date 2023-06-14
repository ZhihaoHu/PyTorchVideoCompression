# FVC: A New Framework towards Deep Video Compression in Feature Space

Official PyTorch implemetation for the paper:

FVC: A New Framework towards Deep Video Compression in Feature Space, Zhihao Hu, Guo Lu, Dong Xu, CVPR 2021 (**Oral**). [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Hu_FVC_A_New_Framework_Towards_Deep_Video_Compression_in_Feature_CVPR_2021_paper.pdf)


## Requirements

- Python==3.6
- PyTorch==1.2

## Data Preparation

### Training data

1. Download [Vimeo-90k dataset](http://toflow.csail.mit.edu/): original training + test set (82GB)

2. Unzip the dataset into `./data/`.

3. Remember to put the file `test.txt` in `./data/vimeo_septuplet/` to the root of your vimeo dataset if you edit the path of vimeo.

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

You can also use BPG or other compression methods for generating better I frames.

## Install Deformable Convolution
    cd subnet
    unzip dcn.zip
    python setup.py develop

## Training
    cd examples/example
    sh cp.sh
    sh run.sh
If you want models with more λ, you can edit `config.json`

If you want to use tensorboard:

    cd examples
    sh tf.sh xxxx

## Testing
Our pretrained models with λ=8192,4096,2048,1024,512,256 are provided on [Google Drive](https://drive.google.com/drive/folders/1sFIsDyiLAcW7CDK65UYYwcF4gGcB4YAV?usp=sharing). You can put it to `snapshot/` and run `test.sh`:

    sh test.sh

As our training strategy is updated, the evaluation results will be better than the results reported in our paper.

## Citation
If you find this paper useful, please cite:
```
@inproceedings{hu2021fvc,
  title={FVC: A new framework towards deep video compression in feature space},
  author={Hu, Zhihao and Lu, Guo and Xu, Dong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1502--1511},
  year={2021}
}
```