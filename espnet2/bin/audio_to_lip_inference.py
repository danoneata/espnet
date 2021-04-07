#!/usr/bin/env python3

"""Audio-to-lip mode decoding."""

import argparse
import logging
import pdb
import shutil
import sys
import time

from pathlib import Path

from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import matplotlib
import numpy as np
import soundfile as sf
import torch
from typeguard import check_argument_types

from espnet.utils.cli_utils import get_commandline_args
from espnet2.fileio.npy_scp import NpyScpWriter
from espnet2.tasks.audio_to_lip import AudioToLip as AudioToLipTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
# from espnet2.audio_to_lip.fastspeech import FastSpeech
from espnet2.utils import config_argparse
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import str2bool
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none


class Audio2Lip:
    """Audio2Lip class

    Examples:
        >>> import soundfile
        >>> audio2lip = Audio2Lip("config.yml", "model.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> audio2lip(audio)

    """
    def __init__(
        self,
        train_config: Optional[Union[Path, str]],
        model_file: Optional[Union[Path, str]] = None,
        dtype: str = "float32",
        device: str = "cpu",
    ):
        assert check_argument_types()

        model, train_args = AudioToLipTask.build_model_from_file(
            train_config, model_file, device
        )
        model.to(dtype=getattr(torch, dtype)).eval()
        self.device = device
        self.dtype = dtype
        self.train_args = train_args
        self.model = model
        self.preprocess_fn = AudioToLipTask.build_preprocess_fn(train_args, False)

    @torch.no_grad()
    def __call__(self, speech):
        assert check_argument_types()

        speech = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))

        batch = {"speech": speech, "speech_lengths": lengths}
        batch = to_device(batch, self.device)

        return self.model.inference(**batch)


def inference(
    output_dir: str,
    batch_size: int,
    dtype: str,
    ngpu: int,
    seed: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    train_config: Optional[str],
    model_file: Optional[str],
):
    """Perform audio-to-lip model decoding."""
    assert check_argument_types()
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build model
    audio2lip = Audio2Lip(
        train_config=train_config,
        model_file=model_file,
        dtype=dtype,
        device=device,
    )

    # 3. Build data-iterator
    loader = AudioToLipTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=AudioToLipTask.build_preprocess_fn(audio2lip.train_args, False),
        collate_fn=AudioToLipTask.build_collate_fn(audio2lip.train_args, False),
        inference=True,
    )

    # 6. Start for-loop
    output_dir = Path(output_dir)
    (output_dir / "lips").mkdir(parents=True, exist_ok=True)
    # (output_dir / "lips_shape").mkdir(parents=True, exist_ok=True)
    # (output_dir / "att_ws").mkdir(parents=True, exist_ok=True)

    with NpyScpWriter(output_dir / "lips", output_dir / "lips/feats.scp") as lips_writer:
        for idx, (keys, batch) in enumerate(loader, 1):
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert _bs == 1, _bs

            # Change to single sequence and remove *_length
            # because inference() requires 1-seq, not mini-batch.
            batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}

            # start_time = time.perf_counter()
            lips = audio2lip(**batch)
            key = keys[0]
            lips_writer[key] = lips.cpu().numpy()


def get_parser():
    """Get argument parser."""
    parser = config_argparse.ArgumentParser(
        description="Audio-to-lip decode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use "_" instead of "-" as separator.
    # "-" is confusing if written in yaml.
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The path of output directory",
    )
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument(
        "--key_file",
        type=str_or_none,
    )

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--train_config",
        type=str,
        help="Training configuration file.",
    )
    group.add_argument(
        "--model_file",
        type=str,
        help="Model parameter file.",
    )

    return parser


def main(cmd=None):
    """Run TTS model decoding."""
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()
