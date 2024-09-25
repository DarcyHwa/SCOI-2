import pickle
import numpy as np
import os


def prepare_dpp_materials(lang, direction="into"):
    bm25_path = f"../data/{lang}/index/test/{direction}/bm25.index"
    
    poly_path = f"../data/{lang}/index/test/{direction}/polynomial.index"
    poly_score_path = f"../data/{lang}/index/test/{direction}/polynomial.score"
    word_vectors_path = f"../data/{lang}/dpp/{direction}/vec.pkl"

    if not all([os.path.exists(path) for path in [poly_path, poly_score_path, word_vectors_path]]):
        raise FileNotFoundError("Some files are missing, make sure you have performed the polynomial.py and dpp_word_diversity.py")
    
    output_path = f"../data/{lang}/dpp/{direction}/dpp_materials.pkl"

    bm25_index = open(bm25_path, "r").read().splitlines()
    bm25_index = [line.split() for line in bm25_index]

    poly_index = open(poly_path, "r").read().splitlines()
    poly_index = [line.split() for line in poly_index]

    poly_score = open(poly_score_path, "r").read().splitlines()
    poly_score = [line.split() for line in poly_score]

    all_word_vectors = pickle.load(open(word_vectors_path, "rb"))

    dpp_materials = []

    for bm25_line, word_vectors, poly_line, poly_score_line in zip(
        bm25_index, all_word_vectors, poly_index, poly_score):

        material = {}
        word_vectors = np.array(word_vectors)

        sim_scores = []
        for index in bm25_line:
            idx = poly_line.index(index)
            score = poly_score_line[idx]
            sim_score = 1 / (1 + float(score))
            sim_scores.append(sim_score)

        material["word_vectors"] = word_vectors
        material["syntax_similarities"] = np.array(sim_scores)
        dpp_materials.append(material)

    with open(output_path, "wb") as f:
        pickle.dump(dpp_materials, f)


if __name__ == "__main__":
    langs = ["de", "fr", "ru"]

    for lang in langs:
        for direction in ["into", "outof"]:
            prepare_dpp_materials(lang, direction)
