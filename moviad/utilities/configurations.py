"""Anomalib library for research and benchmarking."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class LearningType(str, Enum):
    """Learning type defining how the model learns from the dataset samples."""

    ONE_CLASS = "one_class"
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"


class TaskType(str, Enum):
    """Task type used when generating predictions on the dataset."""

    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"

class Split(str, Enum):
    """Dataset split"""

    TRAIN = "train"
    VALID = "valid"
    TEST = "test"

class LabelName(int, Enum):
    """Labels encoding"""

    NORMAL = 0
    ABNORMAL = 1

    