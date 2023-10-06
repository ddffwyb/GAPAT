import numpy as np
import taichi as ti
import taichi_glsl as ts

from .utils import read_config

config = read_config("config/config1.yaml")
vs = config["vs"]
fs = config["fs"]
angle_cos_limit = config["angle_cos_limit"]
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


@ti.kernel
def recon_kernel_das(
    signal_backproj: ti.types.ndarray(),
    detector_location: ti.types.ndarray(),
    signal_recon: ti.types.ndarray(),
    x_start: ti.f32,
    y_start: ti.f32,
    z_start: ti.f32,
):
    for i, j, k in ti.ndrange(num_x, num_y, num_z):
        for n in ti.ndrange(num_detectors):
            dx = x_start + i * res - detector_location[n, 0]
            dy = y_start + j * res - detector_location[n, 1]
            dz = z_start + k * res - detector_location[n, 2]
            d = ts.length(ti.Vector([dx, dy, dz]))
            idx = ti.min(int(d / vs * fs), num_times - 2)
            signal_recon[i, j, k] += signal_backproj[n, idx] * dz / d**3


@ti.kernel
def recon_kernel_angle(
    signal_backproj: ti.types.ndarray(),
    detector_location: ti.types.ndarray(),
    signal_recon: ti.types.ndarray(),
    x_start: ti.f32,
    y_start: ti.f32,
    z_start: ti.f32,
):
    for i, j, k in ti.ndrange(num_x, num_y, num_z):
        for n in ti.ndrange(num_detectors):
            dx = x_start + i * res - detector_location[n, 0]
            dy = y_start + j * res - detector_location[n, 1]
            dz = z_start + k * res - detector_location[n, 2]
            d = ts.length(ti.Vector([dx, dy, dz]))
            angle_cos = dz / d
            idx = ti.min(int(d / vs * fs), num_times - 2) * int(
                angle_cos > angle_cos_limit
            )
            signal_recon[i, j, k] += signal_backproj[n, idx] * angle_cos / d**2


@ti.kernel
def recon_kernel_fbp(
    signal_backproj: ti.types.ndarray(),
    detector_location: ti.types.ndarray(),
    signal_recon: ti.types.ndarray(),
    x_start: ti.f32,
    y_start: ti.f32,
    z_start: ti.f32,
):
    for i, j, k in ti.ndrange(num_x, num_y, num_z):
        for n in ti.ndrange(num_detectors):
            dx = x_start + i * res - detector_location[n, 0]
            dy = y_start + j * res - detector_location[n, 1]
            dz = z_start + k * res - detector_location[n, 2]
            d = ts.length(ti.Vector([dx, dy, dz]))
            angle_cos = dz / d
            idx = ti.min(int(d / vs * fs), num_times - 2) * int(
                angle_cos > angle_cos_limit
            )
            signal_recon[i, j, k] += (
                (
                    signal_backproj[n, idx]
                    - idx * (signal_backproj[n, idx + 1] - signal_backproj[n, idx])
                )
                * angle_cos
                / d**2
            )
