# A comprehensive framework of GPU-accelerated image reconstruction for photoacoustic computed tomography

The repository provides code of the paper with the same name as this repository.

> A comprehensive framework of GPU-accelerated image reconstruction for photoacoustic computed tomography.

## Abstract

**Significance**: Photoacoustic Computed Tomography (PACT) is a promising non-invasive imaging technique for both life science and clinical implementations. To achieve fast imaging speed, modern PACT systems have equipped arrays that have hundreds to thousands of ultrasound transducer (UST) elements, and the element number continues to increase. However, large number of UST elements with parallel data acquisition could generate a massive data size, making it very challenging to realize fast image reconstruction. Although several research groups have developed GPU-accelerated method for PACT, there lacks an explicit and feasible step-by-step description of GPU-based algorithms for various hardware platforms.

**Aim**: In this study, we propose a comprehensive framework for developing GPU-accelerated PACT image reconstruction (Gpu-Accelerated PhotoAcoustic computed Tomography, _**GAPAT**_), helping the research society to grasp this advanced image reconstruction method.

**Approach**: We leverage widely accessible open-source parallel computing tools, including Python multiprocessing-based parallelism, Taichi Lang for Python, CUDA, and possible other backends. We demonstrate that our framework promotes significant performance of PACT reconstruction, enabling faster analysis and real-time applications. Besides, we also described how to realize parallel computing on various hardware configurations, including multicore CPU, single GPU, and multiple GPUs platform.

**Results**: Notably, our framework can achieve an effective rate of approximately 871 times when reconstructing extremely large-scale 3D PACT images on a dual-GPU platform compared to a 24-core workstation CPU. Besides this manuscript, we shared example codes in the GitHub.

**Conclusions**: Our approach allows for easy adoption and adaptation by the research community, fostering implementations of PACT for both life science and medicine.

**Keywords**: photoacoustic computed tomography, large-scale data size, GPU-accelerated method, Taichi Lang for python, multiple GPU platform.

## Installation

The code requires `python>=3.8`, and pre-installing CUDA on your computer is strongly recommended (no specific version requirement). Firstly clone the repository locally.

```bash
git clone https://github.com/ddffwyb/GAPAT.git
cd GAPAT
```

We recommend using the `conda` environment to install the cuda toolkit. The following command installs the cuda toolkit.

```bash
conda create -n gapat python=3.9 -y
conda activate gapat
conda install -c conda-forge cudatoolkit=11.8
```

Then install the dependencies listed in the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## Getting Started

We provide an example of the in vivo experimental by using a synthetic planar array, which is consistent with the experiments and parameters described in Section 3.2 of the paper. In the `config` folder, we place configuration files in `.yaml` format, and by setting the `device` parameter to `gpu` to activate GPU support. In the `data` folder, we provide a set of PACT data of synthetic planar scanning arms. In the `src` folder, we provide the source code of our framework. The `example.py` file provides a demonstration, and the usage method of command line is as follows.

```
python example.py --config_file config.yaml --result_file recon.mat --kernel das
```

## Results

### 240x240x80 Grids

|                      |  CPU  | K-wave CPU | K-wave GPU | 3090 Ti Single | 3090 Ti Dual |
| :------------------: | :---: | :--------: | :--------: | :------------: | :----------: |
|         DAS          | 3034s |   606.6s   |   18.7s    |     13.2s      |    10.8s     |
| DAS with Angle Limit |   /   |     /      |     /      |     13.3s      |    10.9s     |
|         FBP          |   /   |     /      |     /      |     13.8s      |    11.1s     |

### 600x600x200 Grids

|                      |  CPU   | K-wave CPU | K-wave GPU | 3090 Ti Single | 3090 Ti Dual |
| :------------------: | :----: | :--------: | :--------: | :------------: | :----------: |
|         DAS          | 43784s |  6044.7s   |   768.4s   |     100.0s     |    55.0s     |
| DAS with Angle Limit |   /    |     /      |     /      |     100.5s     |    55.4s     |
|         FBP          |   /    |     /      |     /      |     103.7s     |    56.5s     |

### 1200x1200x400 Grids

|                      |   CPU   | K-wave CPU |    K-wave GPU     | 3090 Ti Single | 3090 Ti Dual |
| :------------------: | :-----: | :--------: | :---------------: | :------------: | :----------: |
|         DAS          | 335259s |  44690.4s  | out of GPU memory |     749.7s     |    385.1s    |
| DAS with Angle Limit |    /    |     /      |         /         |     757.3s     |    392.5s    |
|         FBP          |    /    |     /      |         /         |     781.8s     |    402.5s    |
