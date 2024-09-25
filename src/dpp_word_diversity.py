import pickle
from tqdm import tqdm
import os

import numba as nb
import numpy as np

from retriv import SearchEngine

from utils import load_samples, get_collection_data_structure, load_samples_from_index


class Engine(SearchEngine):
    def get_word_vectors(self, query: str):
        query_terms = self.query_preprocessing(query)
        if not query_terms:
            print("No query terms after preprocessing.")
        query_terms = [t for t in query_terms if t in self.vocabulary]
        if not query_terms:
            print("No query terms in vocabulary.")

        doc_ids = self.get_doc_ids(query_terms)
        term_doc_freqs = self.get_term_doc_freqs(query_terms)

        word_vectors = get_word_vectors(
            term_doc_freqs=term_doc_freqs,
            doc_ids=doc_ids,
            relative_doc_lens=self.relative_doc_lens,
            doc_count=self.doc_count,
            **self.hyperparams,
        )

        return word_vectors

def init_engine(name="new-index", training_samples=None):
    assert training_samples is not None, "training_samples is None"

    collection = get_collection_data_structure(training_samples)

    eg = Engine(name, stopwords=None).index(collection, show_progress=False)
    return eg

def get_word_vectors(
    b: float,
    k1: float,
    term_doc_freqs: nb.typed.List[np.ndarray],
    doc_ids: nb.typed.List[np.ndarray],
    relative_doc_lens: nb.typed.List[np.ndarray],
    doc_count: int,
):
    
    word_vectors = np.zeros((doc_count, len(term_doc_freqs)), dtype=np.float32)

    for i in range(len(term_doc_freqs)):
        indices = doc_ids[i]
        freqs = term_doc_freqs[i]

        df = np.float32(len(indices))
        idf = np.float32(np.log(1.0 + (((doc_count - df) + 0.5) / (df + 0.5))))

        word_vectors[indices, i] = idf * (
            (freqs * (k1 + 1.0))
            / (freqs + k1 * (1.0 - b + (b * relative_doc_lens[indices])))
        )

    return word_vectors.tolist()

def index2vec(training_samples, query, name="new-index"):
    engine = init_engine(name=name, training_samples=training_samples)
    word_vectors = engine.get_word_vectors(query)
    return word_vectors

def main(name_prefix, index_path, sample_path, test_data_path, output_path):
    querys = load_samples(test_data_path)

    filtered_training_samples = load_samples_from_index(index_path, sample_path)

    assert len(filtered_training_samples) == len(querys), "len(filtered_training_samples) != len(querys)"

    all_word_vectors = []
    idx = 0
    for query, training_samples in tqdm(zip(querys, filtered_training_samples), total=len(querys)):
        word_vectors = index2vec(training_samples, query, name=name_prefix + str(idx))
        all_word_vectors.append(word_vectors)
        idx += 1

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    with open(output_path, "wb") as f:
        pickle.dump(all_word_vectors, f)
    

if __name__ == "__main__":
    langs = ["de", "fr", "ru"]
    directions = ["into", "outof"]

    for lang in langs:
        for direction in directions:
            name_prefix = f"{lang}-{direction}"
            index_path = f"../data/{lang}/index/test/{direction}/bm25.index"
            output_path = f"../data/{lang}/dpp/{direction}/vec.pkl"
            sample_path = f"../data/{lang}/train.{lang}" if direction == "into" else f"../data/{lang}/train.en"
            test_data_path = f"../data/{lang}/test.{lang}" if direction == "into" else f"../data/{lang}/test.en"

            main(name_prefix, index_path, sample_path, test_data_path, output_path)
        print(f"dpp_word_diversity {lang} done")

    print("dpp_word_diversity all done")
