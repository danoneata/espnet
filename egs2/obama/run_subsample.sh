#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

python=python3       # Specify python to execute espnet commands

. ./path.sh
. ./cmd.sh

todo=
pred_split=
reciprocal=
repeat_num=

. utils/parse_options.sh

data_feats=data

train_set=chunks-train
valid_set=chunks-valid
test_set=chunks-test

suffix="reciprocal-$(printf "%02d" ${reciprocal})-num-${repeat_num}"

logdir="exp/baseline-subsample-v2/${suffix}"
config="exp/baseline-subsample-v2/config.yaml"

train_dir="${data_feats}/${train_set}-${suffix}"
valid_dir="${data_feats}/${valid_set}"
stats_dir="exp/baseline-subsample/${suffix}/stats"

max_epoch=$(echo ${reciprocal} | awk '{printf "%.0f\n", 100 * log($1) / log(2)}')
# warmup_steps=$((5000/${reciprocal}))
# --scheduler_conf "warmup_steps=${warmup_steps}" \

case ${todo} in
    "prepare-folders")
        mkdir -p ${logdir}
        cp exp/baseline-subsample/config.yaml ${logdir}/.
        ;;
    "collect-stats")
        ${python} -m espnet2.bin.audio_to_lip_train \
            --collect_stats true \
            --train_data_path_and_name_and_type "${train_dir}/wav.scp,speech,sound" \
            --train_data_path_and_name_and_type "${train_dir}/lip.scp,lips,npy" \
            --valid_data_path_and_name_and_type "${valid_dir}/wav.scp,speech,sound" \
            --valid_data_path_and_name_and_type "${valid_dir}/lip.scp,lips,npy" \
            --output_dir "${logdir}/stats"
        ;;
    "train")
        model="models/653d10049fdc264f694f57b49849343e/exp/asr_train_asr_transformer_e18_raw_bpe_sp/54epoch.pth"
        # config.yaml is based on the pretrained's model config
        ${python} -m espnet2.bin.audio_to_lip_train \
            --train_data_path_and_name_and_type "${train_dir}/wav.scp,speech,sound" \
            --train_data_path_and_name_and_type "${train_dir}/lip.scp,lips,npy" \
            --train_shape_file "${stats_dir}/train/speech_shape" \
            --train_shape_file "${stats_dir}/train/lips_shape" \
            --valid_data_path_and_name_and_type "${valid_dir}/wav.scp,speech,sound" \
            --valid_data_path_and_name_and_type "${valid_dir}/lip.scp,lips,npy" \
            --valid_shape_file "${stats_dir}/valid/speech_shape" \
            --valid_shape_file "${stats_dir}/valid/lips_shape" \
            --batch_type numel \
            --batch_bins 1000000 \
            --ngpu 1 \
            --init_param ${model}:frontend:frontend ${model}:normalize:normalize ${model}:encoder:encoder \
            --config "${config}" \
            --max_epoch "${max_epoch}" \
            --output_dir "${logdir}/asr"
        ;;
    "infer")
        case ${pred_split} in
            "valid")
                test_dir="${data_feats}/${valid_set}"
                ;;
            "test")
                test_dir="${data_feats}/${test_set}"
                ;;
            *)
                echo "Unknown value for pred_split"
                exit 1
                ;;
        esac
        pred_model_type="ave"
        ${python} -m espnet2.bin.audio_to_lip_inference \
            --ngpu 1 \
            --data_path_and_name_and_type "${test_dir}/wav.scp,speech,sound" \
            --model_file "${logdir}/asr/valid.loss.${pred_model_type}.pth" \
            --train_config "${logdir}/asr/config.yaml" \
            --output_dir "${logdir}/asr/output-${pred_split}-${pred_model_type}"
        ;;
    *)
        echo "Unknown value for todo"
        exit 1
        ;;
esac
