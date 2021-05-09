import argparse
import pdb

from functools import partial

from typing import Callable, Collection, Dict, List, Tuple, Union

import numpy as np
import torch
from typeguard import check_argument_types

from torchvision import transforms as T

import torchvision.models as models  # type: ignore

from espnet2.asr.espnet_model import ESPnetASRMultimodalModel
from espnet2.tasks.asr import ASRTask
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import collate_multimodal_fn


transform_visual = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class EncoderVisual(torch.nn.Module):
    pass


class Resnet18(EncoderVisual):
    def __init__(self):
        super().__init__()
        model = models.resnet18(pretrained=True)
        self.feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))

    def forward(self, x):
        return self.feature_extractor(x)


encoder_visual_choices = ClassChoices(
    name="encoder_visual",
    classes=dict(resnet18=Resnet18),
    type_check=EncoderVisual,
    default="resnet18",
)


class ASRMultimodalTask(ASRTask):
    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        super().add_task_arguments(parser)
        encoder_visual_choices.add_arguments(parser)

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple:
        if not inference:
            retval = ("speech", "visual", "text")
        else:
            # Recognition mode
            retval = ("speech", "visual")
        return retval

    @classmethod
    def build_collate_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        assert check_argument_types()
        # NOTE(kamo): int value = 0 is reserved by CTC-blank symbol
        return partial(
            collate_multimodal_fn,
            float_pad_value=0.0,
            int_pad_value=-1,
            transform_visual=transform_visual,
            not_sequence="visual",
        )

    @classmethod
    def build_preprocess_fn(cls, args: argparse.Namespace, train: bool):
        assert check_argument_types()
        preprocess_fn_speech_text = super().build_preprocess_fn(args, train)
        def preprocess_fn_visual(data):
            data['visual'] = transform_visual(data['visual']).numpy()
            return data
        def preprocess_fn(uid, data):
            data1 = preprocess_fn_speech_text(uid, data)
            data2 = preprocess_fn_visual(data1)
            return data2
        return preprocess_fn

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetASRMultimodalModel:
        asr = super().build_model(args)
        encoder_visual_class = encoder_visual_choices.get_class(args.encoder_visual)
        encoder_visual = encoder_visual_class()
        model = ESPnetASRMultimodalModel(asr=asr, encoder_visual=encoder_visual)
        return model
