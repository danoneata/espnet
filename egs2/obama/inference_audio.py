import pdb
import pickle
import os

import click
import numpy as np
import soundfile
import torch

from espnet2.bin.audio_to_lip_inference import Audio2Lip


PATH_BASE = "models/audio-to-lip-obama"
PATH_PCA = os.path.join(PATH_BASE, "pca-dlib.pkl")
PATH_CONFIG = os.path.join(PATH_BASE, "config.yaml")
PATH_MODEL = os.path.join(PATH_BASE, "valid.loss.ave.pth")

FPS_VIDEO = 29.97
FPS_AUDIO = 16_000


@click.command()
@click.option(
    "-i",
    "--input",
    "path_input",
    required=True,
    type=click.Path(exists=True),
    help="Path to input audio file.",
)
@click.option(
    "-o",
    "--output",
    "path_output",
    required=True,
    type=click.Path(),
    help="Path to predicted lips file.",
)
def main(path_input, path_output):
    audio2lip = Audio2Lip(
        train_config=PATH_CONFIG,
        model_file=PATH_MODEL,
        device="cuda",
        fps_video=FPS_VIDEO,
        fps_audio=FPS_AUDIO,
    )

    with open(PATH_PCA, "rb") as f:
        pca = pickle.load(f)

    audio, rate = soundfile.read(path_input)
    assert rate == FPS_AUDIO

    lips_sm = audio2lip(torch.tensor(audio))
    lips_sm = lips_sm.cpu().numpy().squeeze(0)

    lips_lg = pca.inverse_transform(lips_sm)
    lips_lg = lips_lg.reshape(-1, 20, 2)

    np.save(path_output, lips_lg)


if __name__ == "__main__":
    main()
