# From CTQScorer (Kumar et al., 2023)
# https://github.com/AI4Bharat/CTQScorer/blob/master/dataset/prepare_datasets.sh

# Download datasets - Europarl, Paracrawl
wget https://www.statmt.org/europarl/v10/training/europarl-v10.fr-en.tsv.gz
wget https://www.statmt.org/europarl/v10/training/europarl-v10.de-en.tsv.gz
wget https://s3.amazonaws.com/web-language-models/paracrawl/bonus/en-ru.txt.gz

# Extract data
gzip -d europarl-v10.fr-en.tsv.gz
gzip -d europarl-v10.de-en.tsv.gz
gzip -d en-ru.txt.gz
python extract.py