#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

python=python3       # Specify python to execute espnet commands

. ./path.sh
. ./cmd.sh

asr_type=
model_type=
split=
dset="lrs3"

. utils/parse_options.sh

case $dset in
    "lrs3")
        fps=25.00
        ;;
    "obama-tts")
        fps=29.97
        ;;
    *)
        echo "unknown dataset $dset"
        exit 1
        ;;
esac

logdir=exp/baseline

asr_dir="${logdir}/${asr_type}"
test_dir="data/${dset}-${split}"

${python} -m espnet2.bin.audio_to_lip_inference \
    --ngpu 1 \
    --data_path_and_name_and_type "${test_dir}/wav.scp,speech,sound" \
    --model_file "${asr_dir}/valid.loss.${model_type}.pth" \
    --train_config "${logdir}/config.yaml" \
    --output_dir "${asr_dir}/output-${dset}-${split}-${model_type}" \
    --fps_video ${fps}
