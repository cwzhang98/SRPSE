#!/usr/bin/env bash

DATA_ROOT=$1
TGT_LANG=$2

echo "preparing mustc dataset..."
python examples/speech_to_text/prep_mustc_data.py \
   --data-root "${DATA_ROOT}" --lang "${TGT_LANG}" \
   --vocab-type unigram --vocab-size 10000

echo "generating speaker.txt..."
python MYST/scripts/generate_speakers.py --data-root "${DATA_ROOT}""en-""${TGT_LANG}""/"

echo "audio data augmentation..."
python MYST/scripts/audio_data_aug.py \
--tsv-path "${DATA_ROOT}""en-""${TGT_LANG}""/""train_st.tsv" \
--aug-tsv-path "${DATA_ROOT}""en-""${TGT_LANG}""/""train_st_aug.tsv" \
--aug-dir "${DATA_ROOT}""en-""${TGT_LANG}""/""data_aug" --seed 4398

