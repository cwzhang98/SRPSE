#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

export version="wmt16"
export target=de
DATA_ROOT=$1
echo "arguments:"
echo "DATA_ROOT: $DATA_ROOT"
echo "version: $version"
echo "target language: $target"
echo


mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT" || exit

#echo 'Cloning Moses github repository (for tokenization scripts)...'
#git clone https://github.com/moses-smt/mosesdecoder.git
#echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
#git clone https://github.com/rsennrich/subword-nmt.git

URLS=(
    "http://statmt.org/wmt13/training-parallel-europarl-v7.tgz"
    "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "http://data.statmt.org/wmt16/translation-task/training-parallel-nc-v11.tgz"
    "http://data.statmt.org/wmt16/translation-task/dev.tgz"
    "http://data.statmt.org/wmt16/translation-task/test.tgz"
)
FILES=(
    "training-parallel-europarl-v7.tgz"
    "training-parallel-commoncrawl.tgz"
    "training-parallel-nc-v11.tgz"
    "dev.tgz"
    "test.tgz"
)

orig=orig
mkdir -p $orig
cd $orig || exit

for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}

    if [[ -f $file ]]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS[i]}
        echo "downloading ${file}..."
        wget "$url"
        if [[ -f $file ]]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit -1
        fi
    fi

    if [ "${file: -4}" == ".tgz" ]; then
        tar zxvf "$file"
    elif [ "${file: -4}" == ".tar" ]; then
        tar xvf "$file"
    fi
done