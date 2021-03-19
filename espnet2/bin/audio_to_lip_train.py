#!/usr/bin/env python3

from espnet2.tasks.audio_to_lip import AudioToLip


def get_parser():
    parser = AudioToLip.get_parser()
    return parser


def main(cmd=None):
    r"""Audio-to-lip training.

    Example:
        % python audio_to_lip_train.py x --print_config --optim adadelta \
                > conf/train_audio_to_lip.yaml
        % python audio_to_lip_train.py --config conf/train_audio_to_lip.yaml
    """
    AudioToLip.main(cmd=cmd)


if __name__ == "__main__":
    main()
