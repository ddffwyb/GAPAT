# A-Comprehensive-Framework-for-Developing-GPU-Accelerated-Image-Reconstruction-for-PACT

The repository provides code of the paper with the same name as this repository.

> A Comprehensive Framework for Developing GPU-Accelerated Image Reconstruction for Photoacoustic Computed Tomography.

Photoacoustic Computed Tomography (PACT) is a promising non-invasive imaging technique for both life science and clinical implementations. To achieve fast imaging speed, various arrays that have hundreds to thousands of ultrasound transducer (UST) elements are being installed on the modern PACT systems, and the element number continues to increase. However, large number of UST elements with parallel data acquisition could generate, a massive data size, making it very challenging to realize fast image reconstruction. Although several groups developed GPU-accelerated method for PACT, there lacks an explicit step-by-step description for various hardware platforms. In this study, we propose a comprehensive framework for developing GPU-accelerated PACT image reconstruction, leveraging widely accessible open-source parallel computing tools, including Python multiprocessing-based parallelism, Taichi Lang for Python, CUDA, and possible other backends. We demonstrate that our framework promotes significant performance of PACT reconstruction, enabling faster analysis and real-time applications. Besides, we also described how to realize parallel computing on various hardware configurations, in-cluding multicore CPU, single GPU, and multiple GPUs platform. Notably, our framework can achieve an effective rate of approximately 1000 times when reconstructing extremely large-scale 3D PACT images on a dual-GPU platform compared to a 24-core workstation CPU. Besides this manuscript, we shared example codes in the GitHub. Our approach allows for easy adoption and adaptation by the research community, fostering implementations of PACT for both life science and medicine.

## Installation

The code requires `python>=3.8`, and pre-installing CUDA on your computer is strongly recommended. Firstly clone the repository locally.

```
git clone https://github.com/ddffwyb/A-Comprehensive-Framework-for-Developing-GPU-Accelerated-Image-Reconstruction-for-PACT.git
```

Then install the dependencies listed in the `requirements.txt` file.

```
cd A-Comprehensive-Framework-for-Developing-GPU-Accelerated-Image-Reconstruction-for-PACT
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
