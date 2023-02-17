# An interactive image segmentation application based on S2M

> MiVOS (CVPR 2021) - Scribble To Mask
> [Ho Kei Cheng](https://hkchengrex.github.io/), Yu-Wing Tai, Chi-Keung Tang
> [[arXiv]](https://arxiv.org/abs/2103.07941) [[Paper PDF]](https://arxiv.org/pdf/2103.07941.pdf) [[Project Page]](https://hkchengrex.github.io/MiVOS/)

A simplistic network that turns scribbles to mask. It supports multi-object segmentation using soft-aggregation. Don't expect SOTA results from this model!

![Ex1](https://imgur.com/HesuB4x.gif) ![Ex2](https://imgur.com/NmCrCE1.gif)

## Interactive GUI

`python interactive.py --image <image>`

Controls:

```bash
Mouse Left - Draw scribbles
Mouse middle key - Switch positive/negative
Key f - Commit changes, clear scribbles
Key r - Clear everything
Key d - Switch between overlay/mask view
Key s - Save masks into a temporary output folder (./output/)
```

## Known issues

The model almost always needs to focus on at least one object. It is very difficult to erase all existing masks from an image using scribbles.

## Training

### Datasets

1. Download and extract [LVIS](https://www.lvisdataset.org/dataset) training set.
2. Download and extract [a set of static image segmentation datasets](https://drive.google.com/file/d/1wUJq3HcLdN-z1t4CsUhjeZ9BVDb9YKLd/view?usp=sharing). These are already downloaded for you if you used the `download_datasets.py` in [Mask-Propagation](https://github.com/hkchengrex/Mask-Propagation).

```bash
├── lvis
│   ├── lvis_v1_train.json
│   └── train2017
├── Scribble-to-Mask
└── static
    ├── BIG_small
    └── ...
```

### Commands

Use the `deeplabv3plus_resnet50` pretrained model provided [here](https://github.com/VainF/DeepLabV3Plus-Pytorch).
**You should install Pytorch with cuda at first.**

`CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 9842 --nproc_per_node=2 train.py --id s2m --load_deeplab <path_to_deeplab.pth>`

## Credit

Deeplab implementation and pretrained model: <https://github.com/VainF/DeepLabV3Plus-Pytorch>.

## Citation

Please cite our paper if you find this repo useful!

```bibtex
@inproceedings{cheng2021mivos,
  title={Modular Interactive Video Object Segmentation: Interaction-to-Mask, Propagation and Difference-Aware Fusion},
  author={Cheng, Ho Kei and Tai, Yu-Wing and Tang, Chi-Keung},
  booktitle={CVPR},
  year={2021}
}
```
