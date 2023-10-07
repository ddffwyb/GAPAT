import multiprocessing
import os

import numpy as np
import taichi as ti
import yaml


def read_config(path):
    """
    Reads a YAML configuration file from the specified path and returns its contents as a dictionary.
    """
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def read_all_dat_into_a_matrix(data_path, data_type, num_times, num_channels):
    """
    Read all the .dat file in a directory into a matrix.
    """
    data = []
    dtype = np.int16 if data_type == "int16" else np.float32
    for filename in os.listdir(data_path):
        if filename.endswith(".dat"):
            data.append(
                np.fromfile(os.path.join(data_path, filename), dtype=dtype)
                .reshape(num_times, num_channels)
                .T
            )
    data = np.concatenate(data, axis=0).astype(np.float32)
    # For idx calculating, make idx not out of bounds
    data[:, [0, 1, num_times - 2, num_times - 1]] = 0
    return data


def calculate_detector_location(
    num_detectors, num_channels, detector_interval_x, detector_interval_y
):
    """
    Calculate the location of the detectors.
    """
    detector_location = np.zeros((num_detectors, 3), dtype=np.float32)
    for i in range(num_detectors):
        detector_location[i] = np.array(
            [
                i // num_channels * detector_interval_x,
                i % num_channels * detector_interval_y,
                0,
            ]
        )
    return detector_location


def initialize_signal_recon(num_x, num_y, num_z):
    """
    Initialize the signal reconstruction matrix.
    """
    signal_recon = np.zeros((num_x, num_y, num_z), dtype=np.float32)
    return signal_recon


def recon_single(config, device_no, kernel):
    """
    Implement reconstruction kernel on single CPU or GPU.
    """
    num_channels = config["num_channels"]
    num_steps = config["num_steps"]
    num_times = config["num_times"]
    res = config["res"]
    x_start = config["x_start"]
    x_end = config["x_end"]
    y_start = config["y_start"]
    y_end = config["y_end"]
    z_start = config["z_start"]
    z_end = config["z_end"]
    detector_interval_x = config["detector_interval_x"]
    detector_interval_y = config["detector_interval_y"]
    data_path = config["data_path"]
    data_type = config["data_type"]
    device = config["device"]
    num_devices = config["num_devices"]
    num_detectors = num_channels * num_steps
    num_x = np.around((x_end - x_start) / res).astype(np.int32)
    num_y = np.around((y_end - y_start) / res).astype(np.int32)
    num_z = np.around((z_end - z_start) / num_devices / res).astype(np.int32)
    z = [z_start + i * (z_end - z_start) / num_devices for i in range(num_devices)]
    if device == "cpu":
        ti.init(arch=ti.cpu)
    if device == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_no)
        ti.init(arch=ti.cuda)
    signal_backproj = read_all_dat_into_a_matrix(
        data_path, data_type, num_times, num_channels
    )
    detector_location = calculate_detector_location(
        num_detectors, num_channels, detector_interval_x, detector_interval_y
    )
    signal_recon = initialize_signal_recon(num_x, num_y, num_z)
    kernel(
        signal_backproj, detector_location, signal_recon, x_start, y_start, z[device_no]
    )
    return signal_recon


def recon_multi(config, kernel):
    """
    Implement reconstruction kernel on multiple CPUs or GPUs.
    """
    num_devices = config["num_devices"]
    result = []
    pool = multiprocessing.Pool(processes=num_devices)
    for i in range(num_devices):
        result.append(
            pool.apply_async(
                recon_single,
                (
                    config,
                    i,
                    kernel,
                ),
            )
        )
    pool.close()
    pool.join()
    signal_recon = np.concatenate([result[i].get() for i in range(num_devices)], axis=2)
    return signal_recon
