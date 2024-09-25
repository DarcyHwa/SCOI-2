import numpy as np
import pickle
import os

from dpp_map import fast_map_dpp
from utils import convert_idx


def select(word_vectors, syntax_similarities, max_length=8, scale_factor=1.0) -> list[int]:
    item_size = word_vectors.shape[0]

    syntax_similarities = np.exp(syntax_similarities / (2 * scale_factor))
    word_vectors /= np.linalg.norm(word_vectors, axis=1, keepdims=True) 

    PSD = np.dot(word_vectors, word_vectors.T)
    kernel_matrix = syntax_similarities.reshape((item_size, 1)) * PSD * syntax_similarities.reshape((1, item_size))

    result = fast_map_dpp(kernel_matrix, max_length)
    return result

def main(lang, k=8, output_dir="../data", direction="into", scale_factor=0.5):
    materials_path = f"../data/{lang}/dpp/{direction}/dpp_materials.pkl"

    if not os.path.exists(materials_path):
        raise FileNotFoundError("The DPP materials are missing, make sure you have performed the dpp_materials_preparation.py")

    dpp_materials = pickle.load(open(materials_path, "rb"))

    result = []

    output_dir = f"{output_dir}/{lang}/rerank"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = f"{output_dir}/{direction}/dpp.txt" # Need to be transformed to index file using convert_idx

    for material in dpp_materials:
        word_vectors = material["word_vectors"]
        syntax_similarities = material["syntax_similarities"]
        curr_res = select(word_vectors, syntax_similarities, k, scale_factor)
        result.append(" ".join([str(i) for i in curr_res]))

    with open(output_path, "w") as f:
        f.write("\n".join(result))


if __name__ == "__main__":
    langs = ["de", "fr", "ru"]
    k = 4

    for lang in langs:
        for direction in ["into", "outof"]:
            main(lang, k, direction=direction)
            convert_idx(directions=direction, langs=lang, methods="dpp")
