from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from utils import load_samples, load_samples_from_index


def get_similarity(model, train_data, test_data):
    # Calculate embeddings by calling model.encode()
    train_embeddings = model.encode(train_data)
    test_embedding = model.encode(test_data)

    # Calculate the embedding similarities
    similarities = model.similarity(train_embeddings, test_embedding)
    return similarities.flatten().cpu().numpy()

def get_top_k(similarities, k):
    # Get the top k most similar examples
    top_k = similarities.argsort()[-k:][::-1]
    return top_k

def main(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    langs = ['de', 'fr', 'ru']
    directions = ['into', 'outof']
    split = 'test'

    for lang in langs:
        for direction in directions:
            # Load the pre-trained Sentence Transformer model
            model = SentenceTransformer(model_name)

            # Load the training and test data
            path_to_index = f"../data/{lang}/index/{split}/{direction}/bm25.index"
            path_to_train_data = f"../data/{lang}/train.{lang}" if direction == "into" else f"../data/{lang}/train.en"
            path_to_test_data = f"../data/{lang}/test.{lang}" if direction == "into" else f"../data/{lang}/test.en"
            path_to_output = f"../data/{lang}/index/{split}/{direction}/fuzzy.index"

            train_data, indexes = load_samples_from_index(path_to_index, path_to_train_data, return_indexes=True)
            test_data = load_samples(path_to_test_data)

            # Get the similarity scores
            with open(path_to_output, "w") as f:
                for examples, idxs, test_input in tqdm(zip(train_data, indexes, test_data), total=len(test_data)):
                    similarities = get_similarity(model, examples, test_input)
                    assert len(similarities) == len(examples)
                    top_k = get_top_k(similarities, 8)
                    ranked_indexes = [idxs[i] for i in top_k]

                    f.write(" ".join([str(i) for i in ranked_indexes]) + "\n")


if __name__ == "__main__":
    main()
