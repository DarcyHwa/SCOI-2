import os
import numpy as np
from comet import load_from_checkpoint
from sacrebleu import corpus_bleu


def init_comet_20(model_path="Unbabel/wmt20-comet-da"):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"    
    comet_metric = load_from_checkpoint(model_path)
    return comet_metric


def get_comet_20_scores(predicted, references, source, comet_da_20_metric):
    comet_metric = comet_da_20_metric
    scores = []

    # sometimes we just run for 5 to 10 samples
    k = len(predicted)
    references = references[:k]
    source = source[:k]

    idx = 0
    while idx < len(predicted):
        batch = int(min(1024, len(predicted) - idx))
        predicted_batch = predicted[idx: idx + batch]
        references_batch = references[idx: idx + batch]
        source_batch = source[idx: idx + batch]

        data = []
        for src, mt, ref in zip(source_batch, predicted_batch, references_batch):
            data.append({
                "src": src,
                "mt": mt,
                "ref": ref
            })
        
        comet_score = comet_metric.predict(data, progress_bar=True)        
        scores.extend(comet_score['scores'])
        idx += batch
    
    return scores


def get_comet_20_mean_score(predicted, references, source, comet_da_20_metric):
    scores = get_comet_20_scores(predicted, references, source, comet_da_20_metric)
    mean_score = np.mean(scores)
    mean_score = round(mean_score, 4)
    return mean_score


def get_bleu_scores(predicted, references):
    scores = []
    for pred, refs in zip(predicted, references):
        score = corpus_bleu([pred], [[refs]])
        scores.append(score.score)
    return scores


def get_bleu_mean_score(predicted, references):
    scores = get_bleu_scores(predicted, references)
    mean_score = np.mean(scores)
    mean_score = round(mean_score, 2)
    return mean_score