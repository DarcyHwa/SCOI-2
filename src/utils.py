from typing import List, Union


def load_samples(filepath) -> List[str]:
    return open(filepath, 'r').read().splitlines()


def get_collection_data_structure(samples):
    collection = []
    for id, sample in enumerate(samples):
        current = {"id": id, "text": sample}
        collection.append(current)
    return collection


def load_samples_from_index(index_path, sample_path, return_indexes=False) -> Union[List[List[str]], List[List[int]]]:
    """
    Load samples from a file given an index file.
    
    Args:
        - :param index_path: path to the index file, e.g. "bm25.index"
        - :param sample_path: path to the samples file, e.g. "train.en"
    """

    indexes = open(index_path, 'r').read().splitlines()
    indexes = [line.split() for line in indexes]
    indexes = [[int(i) for i in line] for line in indexes]

    samples = load_samples(sample_path)

    if return_indexes:
        return [[samples[i] for i in line] for line in indexes], indexes
    return [[samples[i] for i in line] for line in indexes]


def convert_idx(directions=["into", "outof"], 
                langs=["de", "fr", "ru"], 
                methods="dpp", 
                data_dir="../data") -> None:
    """
    output_path = f"../data/{lang}/index/test/{direction}/{method}.index"
    """

    if isinstance(methods, str):
        methods = [methods]
    if isinstance(directions, str):
        directions = [directions]
    if isinstance(langs, str):
        langs = [langs]

    for direction in directions:
        for lang in langs:
            for method in methods:
                bm25_path = f"{data_dir}/{lang}/index/test/{direction}/bm25.index"
                relative_idx_path = f"{data_dir}/{lang}/rerank/{direction}/{method}.txt"
                output_path = f"{data_dir}/{lang}/index/test/{direction}/{method}.index"

                indexes = open(bm25_path, "r").read().splitlines()
                indexes = [line.split() for line in indexes]

                relative_idx = open(relative_idx_path, "r").read().splitlines()
                relative_idx = [line.split() for line in relative_idx]
                relative_idx = [[int(idx) for idx in line] for line in relative_idx]

                real_indexes = []
                for relative_idx_line, index_line in zip(relative_idx, indexes):

                    real_index = []
                    for idx in relative_idx_line:
                        real_index.append(index_line[idx])

                    real_indexes.append(real_index)

                with open(output_path, "w") as f:
                    for line in real_indexes:
                        f.write(" ".join(line) + "\n")
