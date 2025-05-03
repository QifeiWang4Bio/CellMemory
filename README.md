# CellMemory

This repository hosts the official implementation of **CellMemory: Hierarchical Interpretation of Out-of-Distribution Cells Using Bottlenecked Transformer**.

<p align="center"><img src="https://github.com/QifeiWang4Bio/CellMemory/blob/main/img/cellmemory.png" alt="cellmemory" width="800px" /></p>

## Installation

To install CellMemory, run the following command:

```bash
conda env create -f environment.yaml
```

[Optional]
```bash
pip install -r requirements.txt
```

## Demo

- Benchmark mHypoMap [script/demo_mHypoMap.ipynb](script/demo_mHypoMap.ipynb)

- CellMemory facilitates the interpretable characterization of single-cell spatial omics [script/demo_hBreastCancer_xenium.ipynb](script/demo_hBreastCancer_xenium.ipynb)

- Hierarchical interpretation of immune cells [script/demo_infer_immunecell.ipynb](script/demo_infer_immunecell.ipynb)

## Citing CellMemory

```bibtex
@article{wang2024cellmemory,
title={CellMemory: Hierarchical Interpretation of Out-of-Distribution Cells Using Bottlenecked Transformer},
author={Wang, Qifei and Zhu, He and Hu, Yiwen and Chen, Yanjie and Wang, Yuwei and Li, Guochao and Li, Yun and Chen, Jinfeng and Zhang, Xuegong and Zou, James and Kellis, Manolis and Li, Yue and Liu, Dianbo and Jiang, Lan},
journal={bioRxiv},
year={2024},
publisher={Cold Spring Harbor Laboratory}
}
```
