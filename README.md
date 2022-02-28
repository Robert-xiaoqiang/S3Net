# S$^3$Net: Self-supervised Self-ensembling Network for Semi-supervised RGB-D Salient Object Detection

- This repository is the official implementation of [the paper](https://ieeexplore.ieee.org/document/9623466).


![](demo/pipeline.png)

## Abstract

RGB-D salient object detection aims to detect visually distinctive objects or regions from a pair of the RGB image and the depth image. State-of-the-art RGB-D saliency detectors are mainly based on convolutional neural networks but almost suffer from an intrinsic limitation relying on the labeled data, thus degrading detection accuracy in complex cases. In this work, we present a self-supervised self-ensembling network (S 3 Net) for semi-supervised RGB-D salient object detection by leveraging the unlabeled data and exploring a self-supervised learning mechanism. To be specific, we first build a self-guided convolutional neural network (SG-CNN) as a baseline model by developing a series of three-layer cross-model feature fusion (TCF) modules to leverage complementary information among depth and RGB modalities and formulating an auxiliary task that predicts a self-supervised image rotation angle. After that, to further explore the knowledge from unlabeled data, we assign SG-CNN to a student network and a teacher network, and encourage the saliency predictions and self-supervised rotation predictions from these two networks to be consistent on the unlabeled data. Experimental results on seven widely-used benchmark datasets demonstrate that our network quantitatively and qualitatively outperforms the state-of-the-art methods.

## Our results

- download from the BaiduPan [link]() or Google Drive [link]().

## Prerequisites

```bash
pip install -r requirements.txt
```

## Datasets

- download all the benchmark datasets of RGB-D saliency from this [link](http://dpfan.net/d3netbenchmark/).
- `unzip` them into the same directory.
- configure the `train/test_datasets_root` items of `code/utils/config.py` using the above directory.   
- download the unlabeled RGB-D datasets from [SUNRGBD](https://rgbd.cs.princeton.edu/) 3D benchmark and discard its semantic labels.

## Train
- configure the `summary_key` item of `code/utils/config.py` for a new experimental run and execute the following scripts.

- Supervised Baseline
```bash
python code/main.py
# which will invoke the trainer of `code/utils/solver.py`.
```

- Semi-supervised Baseline (Vanilla Mean Teacher Framework)
```bash
python code/main_mt.py
# which will invoke the trainer of `code/utils/solver_mt.py`.
```

- Supervised Baseline with Rotation Pretext Learning (Multi-task Learning)
```bash
python code/main_ss.py
# which will invoke the trainer of `code/utils/solver_ss.py`.
```

- Semi-supervised Baseline with Rotation Pretext Learning (**our whole S$^3$Net**)
```bash
python code/main_ss_mt.py
# which will invoke the trainer of `code/utils/solver_ss_mt.py`.
```

## Test

- use the same script as the above for every baseline and adapt the configure `code/utils/config.py` according to your requirements, such as salincy map size and number of the evaluated datasets et al.

## Acknowledge
- thanks to the co-authors for their constructive suggestions.

## License
Copyright 2021 Author of S$^3$Net

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Citation
```latex
@article{zhu2021s,
  title={S $\^{} 3$ Net: Self-supervised Self-ensembling Network for Semi-supervised RGB-D Salient Object Detection},
  author={Zhu, Lei and Wang, Xiaoqiang and Li, Ping and Yang, Xin and Zhang, Qing and Wang, Weiming and Schonlieb, Carola-Bibiane and Chen, CL Philip},
  journal={IEEE Transactions on Multimedia},
  year={2021},
  publisher={IEEE}
}
```