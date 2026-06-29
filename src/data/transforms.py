# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar, Dict, List, Union

import numpy as np
import torch

from scipy import signal

import torchaudio
from dataclasses import field

TTransformIn = TypeVar("TTransformIn")
TTransformOut = TypeVar("TTransformOut")
Transform = Callable[[TTransformIn], TTransformOut]

class DictTransform:
    """
    A base class for transforms that operate on a dictionary of data. The transform is applied to the specified fields in the dictionary.
    """
    def __init__(self, field: Union[str, List[str]] = "emg"):
        self.fields = [field] if isinstance(field, str) else field

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for f in self.fields:
            if f in data:
                data[f] = self.transform(data[f])
        return data

    def transform(self, x: Any) -> Any:
        raise NotImplementedError


@dataclass
class StandardiseSignal(DictTransform):
    """Ensures signal is (C, T) and float32"""
    field: str = "emg"

    def __post_init__(self):
        super().__init__(self.field)

    def transform(self, x: Union[torch.tensor, np.ndarray]) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float()
        if x.shape[0] > x.shape[1]:
            x = x.T
        return x


@dataclass
class PadCropToLength(DictTransform):
    """
    Pad or crop the input tensor along the time dimension to a target length.

    Padding is done on the left side, and cropping is done symmetrically from both sides to preserve the center of the signal.
    """
    field: Union[str, List[str]]
    target_length: int
    pad_value: float = 0.0

    def __post_init__(self):
        super().__init__(self.field)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        current_length = x.shape[1]

        if current_length < self.target_length:
            padding = self.target_length - current_length
            x = torch.nn.functional.pad(x, (padding, 0), value=self.pad_value)
        elif current_length > self.target_length:
            crop_start = (current_length - self.target_length) // 2
            x = x[:, crop_start:crop_start + self.target_length]

        return x


@dataclass
class EncodeStage(DictTransform):
    """
    Specific for emg2pose
    """

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
    field: str = "stage"

    def __post_init__(self):
        super().__init__(self.field)

    def transform(self, x: torch.Tensor | str) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            x = x.item()  # Convert single-element tensor to Python scalar
        
        if x not in self.stage_to_idx:
            raise ValueError(f"Unknown stage: {x}")

        return torch.tensor(
            self.stage_to_idx[x],
            dtype=torch.long
        )


@dataclass
class BandPassFilter(DictTransform):
    lowpass: float
    highpass: float
    order: int
    fs: int
    field: str = "emg"

    def __post_init__(self):
        super().__init__(self.field)
        
        if self.lowpass and self.highpass:
            # Standard Bandpass
            Wn = [self.highpass, self.lowpass]
            btype = "bandpass"
        elif self.lowpass:
            # No highpass -> Lowpass only
            Wn = self.lowpass
            btype = "lowpass"
        elif self.highpass:
            # No lowpass -> Highpass only
            Wn = self.highpass
            btype = "highpass"
        else:
            raise ValueError("At least one cutoff frequency must be provided.")

        b, a = signal.butter(self.order, Wn, btype=btype, fs=self.fs)
        self.b = torch.tensor(b, dtype=torch.float32)
        self.a = torch.tensor(a, dtype=torch.float32)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        # x shape (time, channels)
        squeeze = False
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # (1, channels, time)
            squeeze = True


        y = torchaudio.functional.lfilter(
            x,
            self.a.to(x.device),
            self.b.to(x.device),
        )

        return y.squeeze(0) if squeeze else y


@dataclass
class NotchFilter(DictTransform):
    notch_freq: float
    quality_factor: float
    fs: int
    field: str = "emg"

    def __post_init__(self):
        super().__init__(self.field)
        b, a = signal.iirnotch(self.notch_freq, self.quality_factor, fs=self.fs)
        self.b = torch.tensor(b, dtype=torch.float32)
        self.a = torch.tensor(a, dtype=torch.float32)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        # x shape (time, channels)
        squeeze = False
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # (1, channels, time)
            squeeze = True

        y = torchaudio.functional.lfilter(
            x,
            self.a.to(x.device),
            self.b.to(x.device),
        )

        return y.squeeze(0) if squeeze else y

@dataclass
class Resample(DictTransform):
    orig_fs: int
    target_fs: int
    field: str

    def __post_init__(self):
        super().__init__(self.field)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape C, T OR (B, C, T)
        Returns:
            Tensor with resampled time dimension
        """
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

        return x
    

@dataclass
class RotationAugmentation(DictTransform):
    """Rotate EMG along the channel dimension by a random integer."""
    field: str = "emg"

    def __post_init__(self):
        super().__init__(self.field)

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        rotation = np.random.choice([-1, 0, 1])
        return torch.roll(data, rotation, dims=-1)


@dataclass
class ChannelDownsampling(DictTransform):
    """Downsample number of emg channels."""

    downsampling: int = 2

    def __post_init__(self):
        super().__init__("emg")

    def transform(self, data: torch.Tensor) -> torch.Tensor:
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



@dataclass
class PerChannelZScoreNormalise(DictTransform):
    """Normalise signal to zero mean and unit variance.
    
    Per channel z score normalisation
    """
    field: str = "emg"

    def __post_init__(self):
        super().__init__(self.field)

    def transform(self, x: torch.Tensor) -> torch.Tensor:       
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-6
        return (x - mean) / std
    

@dataclass
class GlobalZScoreNormalise(DictTransform):
    """Normalise signal to zero mean and unit variance.
    
    Global z score normalisation across all channels
    """
    field: str = "emg"

    def __post_init__(self):
        super().__init__(self.field)

    def transform(self, x: torch.Tensor) -> torch.Tensor:       
        mean = x.mean()
        std = x.std() + 1e-6
        return (x - mean) / std


@dataclass
class MinMaxNormalise(DictTransform):
    """Normalise signal to [-1, 1] range."""
    field: str = "emg"

    def __post_init__(self):
        super().__init__(self.field)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        min_val = x.min()
        max_val = x.max()
        return 2 * (x - min_val) / (max_val - min_val + 1e-6) - 1
    

@dataclass
class PeakNormalise(DictTransform):
    """Normalise signal by its peak value."""
    field: str = "emg"

    def __post_init__(self):
        super().__init__(self.field)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        peak = torch.max(torch.abs(x)) + 1e-6
        return x / peak

# Data augmentation transforms
# Similar to TinyMyo

@dataclass
class RandomAmplitudeScale(DictTransform):
    """TinyMyo: Multiplies signal by a random scalar between min and max."""
    min_scale: float = 0.9
    max_scale: float = 1.1
    field: str = "emg"

    def __post_init__(self):
        super().__init__(self.field)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        scale = np.random.uniform(self.min_scale, self.max_scale)
        return x * scale


@dataclass
class RandomChannelDropout(DictTransform):
    """TinyMyo: Completely zeroes out random channels with probability p."""
    dropout_prob: float = 0.05
    field: str = "emg"

    def __post_init__(self):
        super().__init__(self.field)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        # Generate a mask (C, 1) where 1 means keep, 0 means dropout
        mask = torch.rand(x.shape[0], 1, device=x.device) > self.dropout_prob
        return x * mask.float()