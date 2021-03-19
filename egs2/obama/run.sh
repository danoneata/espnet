#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


python=python3       # Specify python to execute espnet commands
logdir=exp/baseline

train_dir=
valid_dir=

. ./path.sh
. ./cmd.sh

data_feats=data/grid

train_set=train
valid_set=valid

train_dir="${data_feats}/${train_set}"
valid_dir="${data_feats}/${valid_set}"

# ${python} -m espnet2.bin.audio_to_lip_train \
#     --collect_stats true \
#     --train_data_path_and_name_and_type "${train_dir}/wav.scp,speech,sound" \
#     --train_data_path_and_name_and_type "${train_dir}/lip.txt,lip_landmarks,rand_float" \
#     --valid_data_path_and_name_and_type "${valid_dir}/wav.scp,speech,sound" \
#     --valid_data_path_and_name_and_type "${valid_dir}/lip.txt,lip_landmarks,rand_float" \
#     --output_dir "${logdir}/stats"
#     # --train_shape_file "${logdir}/train.scp" \
#     # --valid_shape_file "${logdir}/valid.scp" \

stats_dir="exp/baseline/stats"
model="/home/doneata/src/espnet2/tools/venv/lib/python3.8/site-packages/espnet_model_zoo/653d10049fdc264f694f57b49849343e/exp/asr_train_asr_transformer_e18_raw_bpe_sp/54epoch.pth"
# config.yaml is based on the pretrained's model config
${python} -m espnet2.bin.audio_to_lip_train \
    --train_data_path_and_name_and_type "${train_dir}/wav.scp,speech,sound" \
    --train_data_path_and_name_and_type "${train_dir}/lip.txt,lip_landmarks,rand_float" \
    --train_shape_file "${stats_dir}/train/speech_shape" \
    --train_shape_file "${stats_dir}/train/lip_landmarks_shape" \
    --valid_data_path_and_name_and_type "${valid_dir}/wav.scp,speech,sound" \
    --valid_data_path_and_name_and_type "${valid_dir}/lip.txt,lip_landmarks,rand_float" \
    --valid_shape_file "${stats_dir}/valid/speech_shape" \
    --valid_shape_file "${stats_dir}/valid/lip_landmarks_shape" \
    --batch_type numel \
    --batch_bins 20000 \
    --init_param ${model}:frontend:frontend ${model}:normalize:normalize ${model}:encoder:encoder \
    --freeze_param frontend normalize encoder \
    --config exp/baseline/config.yaml \
    --output_dir "${logdir}/stats"
