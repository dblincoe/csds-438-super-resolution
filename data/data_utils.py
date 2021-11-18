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
        lr_image = cv2.resize(
            image,
            (int(image.shape[1] / factor), int(image.shape[0] / factor)))
        lr_images.append(lr_image)

    return lr_images


def resize_images(images: List[np.array], width: int,
                  height: int) -> List[np.array]:
    """Takes list of high-res images and re-shapes them to have the same size

    Args:
        images (List[np.array]): list of images data
        width (int): the new width of the image
        height (int): the new height of the image

    Returns:
        List[np.array]: resized images
    """
    return [cv2.resize(image, (height, width)) for image in images]
