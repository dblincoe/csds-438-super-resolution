from typing import List

import numpy as np
from cv2 import cv2


def downsample_images(images: List[np.array], factor: int) -> List[np.array]:
    """Takes list of high-res images and downsamples them to low-res

    Args:
        images (List[np.array]): list of images data
        factor (int): downsample factor

    Returns:
        List[np.array]: downsampled images
    """
    lr_images = []

    for image in images:
        lr_image = cv2.resize(image, (int(image.shape[1] / factor), int(image.shape[0] / factor)))
        lr_images.append(lr_image)

    return lr_images
