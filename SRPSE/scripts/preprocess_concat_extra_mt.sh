#!/bin/bash
src=$1
tgt=$2
DATA_ROOT=$3 #~/project/dataset/wmt16

SCRIPTS="${HOME}/project/MYST/MYST/scripts/mosesdecoder/scripts"
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
CORPORA=(
    "training-parallel-europarl-v7/europarl-v7.de-en"
    "training-parallel-nc-v11/news-commentary-v11.de-en"
    "commoncrawl/commoncrawl.de-en"
)
cd "$DATA_ROOT" || exit
mkdir -p "en-de"
cd orig || exit

echo "pre-processing train data..."
for lang in $src $tgt; do
  for file in "${CORPORA[@]}" ; do
    cat $file.$lang | perl $NORM_PUNC $lang | perl $REM_NON_PRINT_CHAR >> $DATA_ROOT/en-de/train.$lang
  done
done

echo "pre-processing dev data..."
for lang in $src $tgt; do
    if [ "$lang" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' dev/newstest2015-ende-$t.$lang.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" > $DATA_ROOT/en-de/dev.$lang
done

echo "pre-processing test data..."
# filter text out from .seg file
for lang in $src $tgt; do
    if [ "$lang" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' test/newstest2016-ende-$t.$lang.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" > $DATA_ROOT/en-de/test.$lang
done