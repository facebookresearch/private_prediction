#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Installs dependencies for open-source users.
"""

import os
import requests


# files for RDP dependency:
RDP_BASE = "https://raw.githubusercontent.com/tensorflow/privacy"
RDP_COMMIT = "1ce8cd4032b06e8afa475747a105cfcb01c52ebe"
RDP_FOLDER = "tensorflow_privacy/privacy/analysis"
RDP_FILES = {
    "rdp_accountant.py": "rdp_accountant.py",
    "compute_dp_sgd_privacy.py": "dpsgd_privacy.py"
}

# files for ResNet dependency:
RESNET_BASE = "https://raw.githubusercontent.com/akamaster/pytorch_resnet_cifar10"
RESNET_COMMIT = "4e4f8da1ba2611dad2eedf8a23505c0fbd94b983"
RESNET_FILES = {
    "resnet.py": "resnet.py",
}


def main():

    # download files needed for RDP accountant:
    for source, filename in RDP_FILES.items():
        url = f"{RDP_BASE}/{RDP_COMMIT}/{RDP_FOLDER}/{source}"
        request = requests.get(url, allow_redirects=True)
        with open(filename, "wb") as f:
            print(f"writing file {filename}")
            f.write(request.content)

    # download files needed for ResNet:
    for source, filename in RESNET_FILES.items():
        url = f"{RESNET_BASE}/{RESNET_COMMIT}/{source}"
        request = requests.get(url, allow_redirects=True)
        with open(filename, "wb") as f:
            print(f"writing file {filename}")
            f.write(request.content)

    # apply patches:
    for filename in RDP_FILES.values():
        patch_filename = filename.replace(".py", ".patch")
        os.system(f"patch {filename} patches/{patch_filename}")
    for filename in RESNET_FILES.values():
        patch_filename = filename.replace(".py", ".patch")
        os.system(f"patch {filename} patches/{patch_filename}")
    print("done.")


if __name__ == "__main__":
    main()
