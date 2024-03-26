import argparse
import time

import scipy.io as sio

from src.algorithms import recon_kernel_das, recon_kernel_angle, recon_kernel_fbp
from src.utils import read_config, recon_single, recon_multi

kernel = recon_kernel_das

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reconstruct signal with config file and save the result to specified path"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="config.yaml",
        help="Path to the config file with yaml format",
    )
    parser.add_argument(
        "--result_file",
        type=str,
        default="recon.mat",
        help="Path to save the reconstruction result with mat format",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="das",
        choices=["das", "angle", "fbp"],
        help="Reconstruction kernel, including 'das', 'angle', 'fbp'",
    )
    args = parser.parse_args()

    kernel_options = {
        "das": recon_kernel_das,
        "angle": recon_kernel_angle,
        "fbp": recon_kernel_fbp,
    }

    config_file = args.config_file
    result_file = args.result_file
    kernel = kernel_options[args.kernel]

    config = read_config(config_file)
    num_devices = config["num_devices"]
    start = time.time()
    if num_devices == 1:
        signal_recon = recon_single(config, 0, kernel)
    else:
        signal_recon = recon_multi(config, kernel)
    end = time.time()
    sio.savemat(result_file, {"signal_recon": signal_recon})
    print("Reconstruction time: {:.2f}s".format(end - start))
