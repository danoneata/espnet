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
dset=

. utils/parse_options.sh

case ${dset} in
    trump-chunks-cpac*)
        train_dir=data/${dset}

        valid_set=trump-chunks-corona-valid
        valid_dir=data/${valid_set}

        test_set=trump-chunks-corona-test
        test_dir=data/${test_set}
        ;;
    *)
        echo "Unknown dataset: ${dset}"
        exit 1
        ;;
esac

expdir=exp/finetune/${dset}/asr
asr_stats_dir=exp/finetune/${dset}/stats
predict_output_dir="${expdir}/predict-ave/${test_set}"
config=exp/finetune/config.yaml


case ${todo} in
    "collect-stats")
        ${python} -m espnet2.bin.audio_to_lip_train \
            --collect_stats true \
            --train_data_path_and_name_and_type "${train_dir}/wav.scp,speech,sound" \
            --train_data_path_and_name_and_type "${train_dir}/lip.scp,lips,npy" \
            --valid_data_path_and_name_and_type "${valid_dir}/wav.scp,speech,sound" \
            --valid_data_path_and_name_and_type "${valid_dir}/lip.scp,lips,npy" \
            --output_dir "${asr_stats_dir}"
        ;;
    "train")
        model="exp/baseline/asr-finetune-all/valid.loss.ave.pth"
        # config.yaml is based on the pretrained's model config
        ${python} -m espnet2.bin.audio_to_lip_train \
            --train_data_path_and_name_and_type "${train_dir}/wav.scp,speech,sound" \
            --train_data_path_and_name_and_type "${train_dir}/lip.scp,lips,npy" \
            --train_shape_file "${asr_stats_dir}/train/speech_shape" \
            --train_shape_file "${asr_stats_dir}/train/lips_shape" \
            --valid_data_path_and_name_and_type "${valid_dir}/wav.scp,speech,sound" \
            --valid_data_path_and_name_and_type "${valid_dir}/lip.scp,lips,npy" \
            --valid_shape_file "${asr_stats_dir}/valid/speech_shape" \
            --valid_shape_file "${asr_stats_dir}/valid/lips_shape" \
            --batch_type numel \
            --batch_bins 1000000 \
            --ngpu 1 \
            --init_param ${model} \
            --config "${config}" \
            --output_dir "${expdir}"
        ;;
    "predict")
        ${python} -m espnet2.bin.audio_to_lip_inference \
            --ngpu 1 \
            --data_path_and_name_and_type "${test_dir}/wav.scp,speech,sound" \
            --model_file "${expdir}/valid.loss.ave.pth" \
            --train_config "${expdir}/config.yaml" \
            --output_dir "${predict_output_dir}"
        ;;
    *)
        echo "Unknown todo: ${todo}"
        exit 1
        ;;
esac
