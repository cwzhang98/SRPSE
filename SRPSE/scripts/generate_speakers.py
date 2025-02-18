import argparse
import csv
from tqdm import tqdm

SPLITS = ["train_st", "dev_st", "tst-COMMON_st"]


def main(args):
    speakers = []
    for split in SPLITS:
        with open(args.data_root + split + ".tsv") as f:
            reader = csv.DictReader(f, delimiter="\t", quotechar=None,
                                    doublequote=False, lineterminator="\n",
                                    quoting=csv.QUOTE_NONE)
            for e in tqdm(reader):
                if e["speaker"] not in speakers:
                    speakers.append(e["speaker"])
    with open(args.data_root + "speakers.txt", "w") as speakers_f:
        for speaker in speakers:
            speakers_f.write(speaker + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    args = parser.parse_args()
    main(args)
