## Dependencies
import albumentations as A
from typing import List


## Setup Pipelines

# Define types
train_transform_schedule: List[A.ImageOnlyTransform]
val_transform_schedule: List[A.ImageOnlyTransform]
test_transform_schedule: List[A.ImageOnlyTransform]

# Define the train augmentation pipelines
train_transform_schedule = [
    A.RandomCrop(
        256,
        256,
        p=1,  # <- always apply
    ),  # Randomly crop the image <- choose a random crop of 256x256
    A.HorizontalFlip(p=0.5),  # Randomly flip the image horizontally (50% of the time)
    A.VerticalFlip(p=0.5),  # Randomly flip the image vertically (50% of the time)
    # Apply median blur with a 30% probability, kernes size is 5 <- play with the size to enhance the effect.
    # ADJUST IF, SHOULD OPENCV THROW A WEIRD ERROR.
    # (https://stackoverflow.com/questions/13193207/unsupported-format-or-combination-of-formats-when-using-cvreduce-method-in-ope)
    A.MedianBlur(p=0.3, blur_limit=3),
]

# building blocks for various applications in microscopy
# building blocks = [A.GaussNoise(p=0.5),
#                    A.MedianBlur(p=0.7, blur_limit=(3, 5)),
#                    A.RandomBrightnessContrast(p=0.3, brightness_limit=0.2, contrast_limit=0.2)]


# Define the validation augmentation pipelines
# it is important to have the same transformations for validation
val_transform_schedule = train_transform_schedule

# Define the test augmentation pipelines
# note that we do not want to apply blurring or other soft transformations as we assume peak quality for the test set
# in training, we use blurring to make the model more robust to noise
test_transform_schedule = [
    A.RandomCrop(256, 256, p=1),  # p=1 == apply always
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
]

## Compose the transformations
train_augmentation_pipeline: A.Compose = A.Compose(train_transform_schedule)
val_augmentation_pipeline: A.Compose = A.Compose(val_transform_schedule)
test_augmentation_pipeline: A.Compose = A.Compose(test_transform_schedule)

## Namespace
__all__ = [
    "train_augmentation_pipeline",
    "val_augmentation_pipeline",
    "test_augmentation_pipeline",
]
