from random import shuffle
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from cv2 import cv2
from sklearn.feature_extraction import image as sk_image
from sklearn.utils import gen_batches


def downsample_images(images: List[np.array], factor: int) -> List[np.array]:
    """Takes list of high-res images and downsamples them to low-res

    Args:
        images (List[np.array]): list of images data
        factor (int): downsample factor

    Returns:
        List[np.array]: downsampled images
    """
    lr_patch_size = int(images[0].shape[0] / factor)

    return (
        [ 
            cv2.resize(image, (lr_patch_size, lr_patch_size)) 
            for image in images 
        ]
    )
    


def augment_images(images: List[np.array], patch_size=(100, 100), max_patches=100):
    """Takes list of high-res images and rotates them and patches them

    Args:
        images (List[np.array]): list of images data
        patch_size (Tuple[int,int]): a patch size to extract from the images
        max_patches (int): number of patches to take from each image

    Returns:
        List[np.array]: rotated images and patched images
    """
    if patch_size[0] % 4 != 0:
        raise ValueError("Please set patch size to be multiple of scale value (patch_size/scale = integer)")

    rotated_imgs = np.hstack([_rotate_images(images, i) for i in range(0, 4)])
    return _create_image_patches(
        rotated_imgs, patch_size=patch_size, max_patches=max_patches
    )


def _rotate_images(images: List[np.array], k: int) -> List[np.array]:
    """Takes list of high-res images and rotates them 90*k times

    Args:
        images (List[np.array]): list of images data
        k (int): number of times to rotate 90 degrees

    Returns:
        List[np.array]: rotated images
    """
    return [tf.image.rot90(image, k=k).numpy() for image in images]


def _create_image_patches(images: List[np.array], patch_size, max_patches):
    """Takes list of high-res images and and breaks them into patches of a set patch size

    Args:
        images (List[np.array]): list of images data
        patch_size (Tuple[int,int]): a patch size to extract from the images
        max_patches (int): number of patches to take from each image

    Returns:
        List[np.array]: patches of images
    """
    return np.vstack(
        [
            sk_image.extract_patches_2d(
                image, patch_size=patch_size, max_patches=max_patches, random_state=42
            )
            for image in images
        ]
    )


def split_train_valid_images(
    lr_imgs: List[np.array], hr_imgs: List[np.array], train_test_split=0.8
) -> Tuple[List[np.array], List[np.array]]:
    """Create a train valid split"""
    train_valid_split = int(train_test_split * len(lr_imgs))

    combined_data = list(zip(lr_imgs, hr_imgs))

    train_data = combined_data[:train_valid_split]
    valid_data = combined_data[train_valid_split:]

    return train_data, valid_data


def create_shuffled_batches(train_data: List[np.array], batch_size=10):
    """Shuffle and create batches"""

    # Extract lr and hr images from (lr, hr) pair to (lr images, hr images) batch
    def extract_lr_hr_images_to_batch(x):
        lr_imgs = []
        hr_imgs = []
        for lr_img, hr_img in x:
            lr_imgs.append(lr_img)
            hr_imgs.append(hr_img)
        
        return [np.array(lr_imgs), np.array(hr_imgs)]

    shuffle(train_data)

    batch_slices = gen_batches(len(train_data), batch_size=batch_size)
    return [
        extract_lr_hr_images_to_batch(train_data[batch_slice])
        for batch_slice in batch_slices
    ]
