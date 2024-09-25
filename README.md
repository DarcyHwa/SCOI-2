# SCOI: Syntax-augmented Coverage-based In-context Example Selection for Machine Translation

## Table of Contents
1. [Quick Start](#quick-start)
2. [SCOI Method](#scoi-method)
3. [Citation](#citation)

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
python -m spacy download fr_core_news_sm
python -m spacy download ru_core_news_sm
```

### Preparation
Fetch and extract the training data. Note that test data have been provided for convenience.

```bash
cd data
sh prepare.sh
```

Prepare for in-context examples (except `BM25`, `R-BM25` and `CTQ Scorer`). Note that `BM25`, `R-BM25` and `CTQ Scorer` examples are taken from [AI4Bharat/CTQScorer](https://github.com/AI4Bharat/CTQScorer) since we use the exactly same training and test data.

```bash
cd ../src
sh prepare.sh
```

If you would like to perform the dpp method, run `sh prepare_dpps.sh` also.

### Run Experiments and Evaluation
```bash
sh run.sh
```

## SCOI Method
![SCOI](diagram.svg)

## Citation
If you find our work useful for your research, please cite our paper:
```
@misc{tang2024scoisyntaxaugmentedcoveragebasedincontext,
      title={SCOI: Syntax-augmented Coverage-based In-context Example Selection for Machine Translation}, 
      author={Chenming Tang and Zhixiang Wang and Yunfang Wu},
      year={2024},
      eprint={2408.04872},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.04872}, 
}
```