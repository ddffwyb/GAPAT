# Comprehensive framework of GPU-accelerated image reconstruction for photoacoustic computed tomography

The package `gapat` provides code of the following paper using easy-to-use Python interface with another form and realization of the algorithm.

> Comprehensive framework of GPU-accelerated image reconstruction for photoacoustic computed tomography ([Link](https://www.spiedigitallibrary.org/journals/journal-of-biomedical-optics/volume-29/issue-06/066006/Comprehensive-framework-of-GPU-accelerated-image-reconstruction-for-photoacoustic-computed/10.1117/1.JBO.29.6.066006.full#_=_))

## Abstract

**Significance**: Photoacoustic Computed Tomography (PACT) is a promising non-invasive imaging technique for both life science and clinical implementations. To achieve fast imaging speed, modern PACT systems have equipped arrays that have hundreds to thousands of ultrasound transducer (UST) elements, and the element number continues to increase. However, large number of UST elements with parallel data acquisition could generate a massive data size, making it very challenging to realize fast image reconstruction. Although several research groups have developed GPU-accelerated method for PACT, there lacks an explicit and feasible step-by-step description of GPU-based algorithms for various hardware platforms.

**Aim**: In this study, we propose a comprehensive framework for developing GPU-accelerated PACT image reconstruction (Gpu-Accelerated PhotoAcoustic computed Tomography, _**GAPAT**_), helping the research society to grasp this advanced image reconstruction method.

**Approach**: We leverage widely accessible open-source parallel computing tools, including Python multiprocessing-based parallelism, Taichi Lang for Python, CUDA, and possible other backends. We demonstrate that our framework promotes significant performance of PACT reconstruction, enabling faster analysis and real-time applications. Besides, we also described how to realize parallel computing on various hardware configurations, including multicore CPU, single GPU, and multiple GPUs platform.

**Results**: Notably, our framework can achieve an effective rate of approximately 871 times when reconstructing extremely large-scale 3D PACT images on a dual-GPU platform compared to a 24-core workstation CPU. Besides this manuscript, we shared example codes in the GitHub.

**Conclusions**: Our approach allows for easy adoption and adaptation by the research community, fostering implementations of PACT for both life science and medicine.

**Keywords**: photoacoustic computed tomography, large-scale data size, GPU-accelerated method, Taichi Lang for python, multiple GPU platform.

## Installation

The code requires python>=3.8, and pre-installing CUDA on your computer is strongly recommended (no specific version requirement).

We recommend using the conda environment to install the cuda toolkit. The following command installs the cuda toolkit.

```bash
conda create -n gapat python=3.9 -y
conda activate gapat
conda install -c conda-forge cudatoolkit=11.8
```

Then install the package using pip.

```bash
pip install gapat
```

## Getting Started

Here is a simple example to show how to use the package. You can also read the following documentation for more details.

```python
import time

import numpy as np
from gapat.algorithms import recon
from gapat.processings import negetive_processing
from gapat.utils import load_dat, calculate_detector_location, save_mat

# Settings
data_path = "./data"
num_channels = 256
num_times = 2048
detector_interval_x = 0.10e-3  # Scanning spacing
detector_interval_y = 0.50e-3  # Linear array spacing
x_range = [-10.0e-3, 110.0e-3]
y_range = [-10.0e-3, 110.0e-3]
z_range = [10.0e-3, 50.0e-3]
res = 0.20e-3
vs = 1500.0
fs = 40.0e6
delay = 0
method = "das"
device = "gpu"
num_devices = 1
device_no = 0

# Load data
signal_backproj = load_dat(data_path, num_channels, num_times)
num_detectors = signal_backproj.shape[0]
detector_location = calculate_detector_location(num_detectors, num_channels, detector_interval_x, detector_interval_y)
detector_normal = np.zeros((num_detectors, 3), dtype=np.float32)
detector_normal[:, 2] = 1.0  # The normal of the detectors points to the volume of planar array

# Reconstruction
start = time.time()
signal_recon = recon(
    signal_backproj,
    detector_location,
    detector_normal,
    x_range,
    y_range,
    z_range,
    res,
    vs,
    fs,
    delay,
    method,
    device,
    num_devices,
    device_no,
)
end = time.time()
print(f"Reconstruction time: {end - start} seconds")

# Save the result
signal_recon = negetive_processing(signal_recon, method="zero", axis=0)
save_mat("result.mat", "signal_recon", signal_recon)
```

## Documentation

### gapat.algorithms

#### gapat.algorithms.recon(signal_backproj, detector_location, detector_normal, x_range, y_range, z_range, res, vs, fs, delay=0, method="das", device="gpu", num_devices=1, device_no=0, block_dim=512)

Reconstruction of photoacoustic computed tomography.

**Warning**: When using multi-device reconstruction, the function must be called on the main process.

**Parameters**

| Parameter           | Type         | Description                                                                                                                                     |
| ------------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `signal_backproj`   | `np.ndarray` | The input signal. Each row is a signal of a detector.<br>Shape: (num_detectors, num_times). Dtype: np.float32.                                  |
| `detector_location` | `np.ndarray` | The location of the detectors. Each row is the coordinates of a detector.<br>Shape: (num_detectors, 3). Dtype: np.float32.                      |
| `detector_normal`   | `np.ndarray` | The normal of the detectors. Each row is the normal of a detector which points to the volume.<br>Shape: (num_detectors, 3). Dtype: np.float32.  |
| `x_range`           | `list`       | The range of the reconstruction volume. The first is the start x and the second is the end x. Example: [0, 1].<br>Shape: (2,). Dtype: float.    |
| `y_range`           | `list`       | The range of the reconstruction volume. The first is the start y and the second is the end y. Example: [0, 1].<br>Shape: (2,). Dtype: float.    |
| `z_range`           | `list`       | The range of the reconstruction volume. The first is the start z and the second is the end z. Example: [0, 1].<br>Shape: (2,). Dtype: float.    |
| `res`               | `float`      | The resolution of the volume.                                                                                                                   |
| `vs`                | `float`      | The speed of sound in the volume.                                                                                                               |
| `fs`                | `float`      | The sampling frequency.                                                                                                                         |
| `delay`             | `int`        | The delay of the detectors. Default: 0.                                                                                                         |
| `method`            | `str`        | The method to use. Default: "das". Options: "das", "ubp".                                                                                       |
| `device`            | `str`        | The device to use. Default: "gpu". Options: "cpu", "gpu".                                                                                       |
| `num_devices`       | `int`        | The number of devices to use. When = 1, the device set by `device_no` will be used.<br>When \> 1, the first n devices will be used. Default: 1. |
| `device_no`         | `int`        | The device no to use when num_devices = 1. Default: 0.                                                                                          |
| `block_dim`         | `int`        | The block dimension. Default: 512.                                                                                                              |

**Returns**

| Parameter      | Type         | Description                                                                   |
| -------------- | ------------ | ----------------------------------------------------------------------------- |
| `signal_recon` | `np.ndarray` | The reconstructed signal.<br>Shape: (num_x, num_y, num_z). Dtype: np.float32. |

### gapat.processings

#### gapat.processings.bandpass_filter(signal_matrix, fs, band_range, order=2, axis=0)

Bandpass filter the signal matrix.

**Parameters**

| Parameter       | Type         | Description                                                                                                                                               |
| --------------- | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `signal_matrix` | `np.ndarray` | The signal matrix to be filtered.<br>Shape: (num_detectors, num_times). Dtype: np.float32.                                                                |
| `fs`            | `float`      | The sampling frequency (Hz).                                                                                                                              |
| `band_range`    | `list`       | The band range to filter (Hz). The first is the low frequency and the second is the high frequency. Example: [10e6, 100e6].<br>Shape: (2,). Dtype: float. |
| `order`         | `int`        | The order of the filter. Default: 2.                                                                                                                      |
| `axis`          | `int`        | The axis to filter. Default: 0. (Which will be applied to each detector.)                                                                                 |

**Returns**

| Parameter                | Type         | Description                                                                          |
| ------------------------ | ------------ | ------------------------------------------------------------------------------------ |
| `filtered_signal_matrix` | `np.ndarray` | The filtered signal matrix.<br>Shape: (num_detectors, num_times). Dtype: np.float32. |

#### gapat.processings.negetive_processing(signal_recon, method="zero", axis=0)

Process the negative signal.

**Parameters**

| Parameter      | Type         | Description                                                                                                                                                                                                                                                                     |
| -------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `signal_recon` | `np.ndarray` | The reconstructed signal to be processed.<br>Shape: (num_x, num_y, num_z). Dtype: np.float32.                                                                                                                                                                                   |
| `method`       | `str`        | The method to process the negative signal. Default: "zero". Options: "zero", "abs", "hilbert".<br>"zero": Set the negative signal to zero.<br>"abs": Take the absolute value of the negative signal.<br>"hilbert": Use the hilbert transform to get the envelope of the signal. |
| `axis`         | `int`        | The axis to process when method is "hilbert". Default: 0.                                                                                                                                                                                                                       |

**Returns**

| Parameter                | Type         | Description                                                                              |
| ------------------------ | ------------ | ---------------------------------------------------------------------------------------- |
| `processed_signal_recon` | `np.ndarray` | The processed signal reconstruction.<br>Shape: (num_x, num_y, num_z). Dtype: np.float32. |

### gapat.utils

#### gapat.utils.load_mat(filename)

Load .mat file and return a dictionary with variable names as keys, and loaded matrices as values.

**Parameters**

| Parameter  | Type  | Description                |
| ---------- | ----- | -------------------------- |
| `filename` | `str` | The path to the .mat file. |

**Returns**

| Parameter | Type   | Description                                                              |
| --------- | ------ | ------------------------------------------------------------------------ |
| `data`    | `dict` | A dictionary with variable names as keys, and loaded matrices as values. |

#### gapat.utils.save_mat(filename, varname, data)

Save data to .mat file with the given variable name.

**Parameters**

| Parameter  | Type         | Description                            |
| ---------- | ------------ | -------------------------------------- |
| `filename` | `str`        | The path to the .mat file.             |
| `varname`  | `str`        | The variable name to save the data to. |
| `data`     | `np.ndarray` | The data to save.                      |

#### gapat.utils.load_dat(data_path, num_channels, num_times, dtype=np.int16, order="F", zero_set=True)

Load all .dat files in the given directory and return a numpy array.

**Parameters**

| Parameter      | Type       | Description                                                                                                      |
| -------------- | ---------- | ---------------------------------------------------------------------------------------------------------------- |
| `data_path`    | `str`      | The path to the .dat files.                                                                                      |
| `num_channels` | `int`      | The number of channels.                                                                                          |
| `num_times`    | `int`      | The number of times.                                                                                             |
| `dtype`        | `np.dtype` | The data type of the data needed to be loaded. Default: np.int16.                                                |
| `order`        | `str`      | The order of the loaded data. Same as numpy.reshape order. "F" for Fortran order, "C" for C order. Default: "F". |
| `zero_set`     | `bool`     | Whether to set the first and last two rows to 0.0. Default: True.                                                |

**Returns**

| Parameter | Type         | Description                                                                           |
| --------- | ------------ | ------------------------------------------------------------------------------------- |
| `data`    | `np.ndarray` | The loaded data.<br>Shape: (num_channels \* num_files, num_times). Dtype: np.float32. |

#### gapat.utils.calculate_detector_location(num_detectors, num_channels, detector_interval_x, detector_interval_y)

Calculate the location of the detectors.

**Parameters**

| Parameter             | Type    | Description                                                                |
| --------------------- | ------- | -------------------------------------------------------------------------- |
| `num_detectors`       | `int`   | The number of total detectors. Equivalent to num_channels \* num_steps.    |
| `num_channels`        | `int`   | The number of channels.                                                    |
| `detector_interval_x` | `float` | The interval of the detectors in the x direction (scanning direction).     |
| `detector_interval_y` | `float` | The interval of the detectors in the y direction (linear array direction). |

**Returns**

| Parameter           | Type         | Description                                                                     |
| ------------------- | ------------ | ------------------------------------------------------------------------------- |
| `detector_location` | `np.ndarray` | The location of the detectors.<br>Shape: (num_detectors, 3). Dtype: np.float32. |
