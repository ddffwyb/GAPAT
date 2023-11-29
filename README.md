# A comprehensive framework of GPU-accelerated image reconstruction for photoacoustic computed tomography

The repository provides code of the paper with the same name as this repository.

> A comprehensive framework of GPU-accelerated image reconstruction for photoacoustic computed tomography.

Photoacoustic Computed Tomography (PACT) is a promising non-invasive imaging technique for both life science and clinical implementations. To achieve fast imaging speed, various arrays that have hundreds to thousands of ultrasound transducer (UST) elements are being installed on the modern PACT systems, and the element number continues to increase. However, large number of UST elements with parallel data acquisition could generate, a massive data size, making it very challenging to realize fast image reconstruction. Although several groups have developed GPU-accelerated method for PACT, there lacks an explicit step-by-step description for various hardware platforms. In this study, we propose a comprehensive framework for developing GPU-accelerated PACT image reconstruction, leveraging widely accessible open-source parallel computing tools, including Python multiprocessing-based parallelism, Taichi Lang for Python, CUDA, and possible other backends. We demonstrate that our framework promotes significant performance of PACT reconstruction, enabling faster analysis and real-time applications. Besides, we also described how to realize parallel computing on various hardware configurations, in-cluding multicore CPU, single GPU, and multiple GPUs platform. Notably, our framework can achieve an effective rate of approximately 871 times faster when reconstructing large-scale 3D PACT images on a dual-GPU platform compared to a 24-core workstation CPU. By optimizing the organization of data, refining the indexing of the DAS algorithm, and setting precompiled pa-rameters, we maximize the utilization and occupancy of the GPU as well as multi-GPU configu-rations, among others. Besides this manuscript, we uploaded example source codes to the GitHub for everyone to learn and use. Our approach allows for easy adoption and adaptation by the re-search community, fostering implementations of PACT for both life science and medicine.

## Installation

The code requires `python>=3.8`, and pre-installing CUDA on your computer is strongly recommended. Firstly clone the repository locally.

```
git clone https://github.com/ddffwyb/A-comprehensive-framework-of-GPU-accelerated-image-reconstruction-for-PACT.git
```

Then install the dependencies listed in the `requirements.txt` file.

```
cd A-comprehensive-framework-of-GPU-accelerated-image-reconstruction-for-PACT
pip install -r requirements.txt
```

The following optional dependencies are necessary for running the example notebooks.

```
pip install jupyter matplotlib
```

## Getting Started

We provide an example of the in vivo experimental by using a synthetic planar array, which is consistent with the experiments and parameters described in Section 3.2 of the paper. In the `config` folder, we place configuration files in `.yaml` format. In the `data` folder, we provide a set of PACT data of synthetic planar scanning arms. In the `src` folder, we provide the source code of our framework. The `example.py` file provides a demonstration, and the usage method of command line is as follows.

```
python example.py --config_path config/config1.yaml --result_path recon.mat
```

## Results

### 240x240x80 Grids

|                |  CPU  | K-wave | 3090 Ti Single | 3090 Ti Dual |
| :------------: | :---: | :----: | :------------: | :----------: |
|      DAS       | 3034s | 606.6s |     13.2s      |    10.8s     |
| DAS with Angle |   /   |   /    |     13.3s      |    10.9s     |
|      FBP       |   /   |   /    |     13.8s      |    11.1s     |

### 600x600x200 Grids

|                |  CPU   | K-wave  | 3090 Ti Single | 3090 Ti Dual |
| :------------: | :----: | :-----: | :------------: | :----------: |
|      DAS       | 43784s | 6044.7s |     100.0s     |    55.0s     |
| DAS with Angle |   /    |    /    |     100.5s     |    55.4s     |
|      FBP       |   /    |    /    |     103.7s     |    56.5s     |

### 1200x1200x400 Grids

|                |   CPU   |  K-wave  | 3090 Ti Single | 3090 Ti Dual |
| :------------: | :-----: | :------: | :------------: | :----------: |
|      DAS       | 335259s | 44690.4s |     749.7s     |    385.1s    |
| DAS with Angle |    /    |    /     |     757.3s     |    392.5s    |
|      FBP       |    /    |    /     |     781.8s     |    402.5s    |
