#!/usr/bin/env python3
from espnet2.tasks.asr_multimodal import ASRMultimodalTask


def get_parser():
    parser = ASRMultimodalTask.get_parser()
    return parser


def main(cmd=None):
    r"""Multimodal ASR training.

    Example:

        % python asr_multimodal_train.py asr --print_config --optim adadelta \
                > conf/train_asr_multimodal.yaml
        % python asr_multimodal_train.py --config conf/train_asr_multimodal.yaml
    """
    ASRMultimodalTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
