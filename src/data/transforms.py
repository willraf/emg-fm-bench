# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import torch

from scipy import signal

import torchaudio
from dataclasses import field


TTransformIn = TypeVar("TTransformIn")
TTransformOut = TypeVar("TTransformOut")
Transform = Callable[[TTransformIn], TTransformOut]


class DictTransform:
    pass


class TensorTransform:
    pass


@dataclass
class PadCropToLength(TensorTransform):
    """
    Pad or crop the input tensor along the time dimension to a target length.

    Padding is done on the left side, and cropping is done symmetrically from both sides to preserve the center of the signal.
    """

    target_length: int
    pad_value: float = 0.0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        current_length = x.shape[0]
        print("--------PadCropToLength--------")
        print(f"Start shape: {x.shape}")

        if current_length < self.target_length:
            padding = self.target_length - current_length
            x = torch.nn.functional.pad(x, (0, 0, padding, 0), value=self.pad_value)
        elif current_length > self.target_length:
            crop_start = (current_length - self.target_length) // 2
            x = x[crop_start:crop_start + self.target_length]

        print(f"End shape: {x.shape}")
        return x


@dataclass
class EncodeStage(DictTransform):
    stage_field: str = "stage"
    stage_to_idx: dict = field(default_factory=lambda: {
        'ThumbsUpDownThumbRotationsCWCCWP': 0,
        'HandClawGraspFlicks': 1,
        'ShakaVulcanPeace': 2,
        'HookEmHornsOKScissors': 3,
        'TwoHandedFreeStyle': 4,
        'unconstrained': 5,
        'OneHandedFreeStyle': 6,
        'CountingUpDownFaceSideAway': 7,
        'ThumbsSwipesWholeHand': 8,
        'FingerWigglingSpreading': 9,
        'FingerAbductionSeries': 10,
        'FingerPinchesSingleFingerPinchesMultiple': 11,
        'FingerFreeform': 12,
        'IndividualFingerPointingSnap': 13,
        'GraspPunchCloseFar': 14,
        'PokeDrawPinchRotateclosefar': 15,
        'AllFingerPinchesThumbSwipeThumbRotate': 16,
        'CountingUpDownFingerWigglingSpreading': 17,
        'IndexPinchesMiddlePinchesThumbswipes': 18,
        'DoorknobFingerGraspFistGrab': 19,
        'FastPongFronthandBackhandThrowing': 20,
        'HandDeskSeparateClaspedChest': 21,
        'PokePinchCloseFar': 22,
        'FingerTouchPalmClapmrburns': 23,
        'WristFlexionAbduction': 24,
        'PlayBlocksChess': 25,
        'CoffeePanicPete': 26,
        'HandOverHandAllFingerPinchesThumbSwipeThumbRotate': 27,
        'HandOverHandCountingUpDownFingerWigglingSpreading': 28
    })

    def __call__(self, data: dict) -> dict:
        stage = data[self.stage_field]
        data[self.stage_field] = torch.tensor(
            self.stage_to_idx[stage],
            dtype=torch.long
        )
        return data


@dataclass
class NeuroRVQEmbeddings(DictTransform):
    """Adds spacial and temporal embeddings to the data dict, using neuroRVQ method

    Args:
        n_patches (int): The maximum number of patches to be extracted from the data. This is determined by the max number the model can process

    
    """

    n_patches: int
    channel_field: str = "channels"
    global_channels: tuple = (
        'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8',
        'c9', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15', 'c16'
    )
    # Optionally add the sample channels, if all samples are going to be the same
    sample_channels: int = None

    def __post_init__(self):
        # Convert to numpy array for efficient lookups
        self._global_channels_array = np.array([
            c.encode() if isinstance(c, str) else c 
            for c in self.global_channels
        ])

        if self.sample_channels is not None:
            self._sample_channels_array = np.array([
                c.encode() if isinstance(c, str) else c 
                for c in self.sample_channels
            ])
    
    def __call__(self, data: dict) -> dict:
        if self.sample_channels is not None:
            sample_channels = self._sample_channels_array
        else:
            sample_channels = data[self.channel_field]

        # Ensure channels are in bytes format for comparison
        if isinstance(sample_channels, (list, np.ndarray)):
            sample_channels = np.array([
                c.encode() if isinstance(c, str) else c 
                for c in sample_channels
            ])

        
        n_channels = len(sample_channels)
        # The number of patches in the time dimension
        # for aligning the current time-point to the right.
        n_time = self.n_patches // n_channels
    

        # temporal
        temp_ix = torch.arange(
            self.n_patches - n_time,
            self.n_patches
        ).repeat(len(sample_channels))

        # spatial
        spat_ix = torch.tensor([
            np.where(self._global_channels_array == c)[0][0]
            for c in sample_channels
        ])

        spat_ix = torch.repeat_interleave(spat_ix, n_time)
        spat_ix = spat_ix

        data['temporal_ix'] = temp_ix
        data['spatial_ix'] = spat_ix
        return data


@dataclass
class BandPassFilter(TensorTransform):
    lowpass: float
    highpass: float
    order: int
    fs: int

    def __post_init__(self):
        b, a = signal.butter(
            self.order,
            [self.highpass, self.lowpass],
            btype="bandpass",
            fs=self.fs,
        )

        self.b = torch.tensor(b, dtype=torch.float32)
        self.a = torch.tensor(a, dtype=torch.float32)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        print("--------BandPassFilter--------")
        print(f"Start shape: {x.shape}")
        # x shape (time, channels)
        x = x.T.unsqueeze(0)  # (1, channels, time)

        y = torchaudio.functional.lfilter(
            x,
            self.a.to(x.device),
            self.b.to(x.device),
        )
        print(f"End shape: {y.squeeze(0).T.shape}")

        return y.squeeze(0).T


@dataclass
class Resample(TensorTransform):
    orig_fs: int
    target_fs: int

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (T, C) or (B, T, C)
        Returns:
            Tensor with resampled time dimension
        """
        print("--------Resample--------")
        print(f"Start shape: {x.shape}")

        x = x.T

        if self.orig_fs == self.target_fs:
            return x

        squeeze = False
        if x.ndim == 2:  # (C, T)
            x = x.unsqueeze(0)
            squeeze = True

        B, C, T = x.shape
        x = x.reshape(B * C, T)

        x = torchaudio.functional.resample(
            x,
            orig_freq=self.orig_fs,
            new_freq=self.target_fs
        )

        x = x.reshape(B, C, -1)

        if squeeze:
            x = x.squeeze(0)

        print(f"End shape: {x.T.shape}")

        return x.T


@dataclass
class ExtractToTensor(TensorTransform):
    """Extracts the specified ``fields`` from a numpy structured array
    and stacks them into a ``torch.Tensor``.

    Following TNC convention as a default, the returned tensor is of shape
    (time, field/batch, electrode_channel).

    Args:
        fields (list): List of field names to be extracted from the passed in
            structured numpy ndarray.
        stack_dim (int): The new dimension to insert while stacking
            ``fields``. (default: 1)
    """

    field: str = "emg"

    def __call__(self, data: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(data[self.field])


@dataclass
class RotationAugmentation(TensorTransform):
    """Rotate EMG along the channel dimension by a random integer."""

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        rotation = np.random.choice([-1, 0, 1])
        return torch.roll(data, rotation, dims=-1)


@dataclass
class ChannelDownsampling(TensorTransform):
    """Downsample number of emg channels."""

    downsampling: int = 2

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data[:, :: self.downsampling]


@dataclass
class Compose:
    """Compose a chain of transforms.

    Args:
        transforms (list): List of transforms to compose.
    """

    transforms: Sequence[Transform[Any, Any]]

    def __call__(self, data: Any) -> Any:
        for transform in self.transforms:
            data = transform(data)
        return data