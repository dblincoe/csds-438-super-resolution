import os
from typing import List
from cv2 import cv2

import numpy as np

root_path = "/".join(os.path.abspath(__file__).split("/")[:-2])

def load_test_images() -> List[np.array]:
    """Load specifically files in 'test-images'

    Returns:
        List[np.array]: List of test images data
    """
    return load_images_from_folder("test-images")


def load_images_from_folder(folder_name: str) -> List[np.array]:
    """Loads images from a folder

    Args:
        folder_name (str): relative folder to load images from

    Returns:
        List[np.array]: List of images data
    """
    absolute_folder_name = os.path.join(root_path, folder_name)

    images = []
    for filename in os.listdir(absolute_folder_name):
        img = cv2.imread(os.path.join(absolute_folder_name, filename))
        if img is not None:
            images.append(img)
    return images
