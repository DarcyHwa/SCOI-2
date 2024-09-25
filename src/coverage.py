import json
import numpy as np
import random
import time

from tqdm import tqdm


lang2nlabels = {
    'en': 45,
    'de': 42,
    'fr': 36,
    'ru': 41,
}


class Node:
    def __init__(self, idx, head_idx, label_idx):
        self.idx = idx
        self.head = head_idx
        self.label = label_idx
        self.children = []
    
    def add_child(self, child):
        self.children.append(child)

    def del_child(self, child):
        self.children.remove(child)


def read_tree_file(tree_fn, n_dim):
    with open(tree_fn, 'r') as f:
        trees = json.load(f)
    
    polynomials = []

    for tree in tqdm(trees, ncols=60):
        nodes = []
        nodes.append(Node(0, -1, 0))

        for node in tree['tree']:
            nodes.append(Node(node[0], node[1], node[2]))

        for i in range(1, len(nodes)):
            nodes[nodes[i].head].add_child(nodes[i])

        polynomial = tree2polynomial(nodes[0], n_dim)
        polynomials.append(polynomial)
    
    return polynomials


def read_moses_file(moses_fn):
    tokenized_sentences = []

    with open(moses_fn, 'r') as f:
        for line in f:
            tokenized_sentences.append(line.strip().split())

    return tokenized_sentences


# returns a polynomial
# each polynomial contains a list of terms
def tree2polynomial(root: Node, n_dim: int):
    if is_leaf(root):
        term = np.zeros(n_dim, dtype=np.int16)
        term[root.label] = 1
        return np.array([term], dtype=np.int16)
    
    first_child = True
    for child in root.children:
        child_poly = tree2polynomial(child, n_dim)
        if first_child:
            poly = child_poly
            first_child = False
        else:
            poly = polynomial_mul(poly, child_poly)
    
    term = np.zeros(n_dim, dtype=np.int16)
    term[root.label] = 1
    poly = polynomial_time(poly, term)

    term = np.zeros(n_dim, dtype=np.int16)
    term[root.label] = 1
    poly = np.vstack((poly, term))
    
    return poly


def polynomial_mul(poly1, poly2):
    return np.vstack((poly1, poly2))


def polynomial_time(poly, term):
    return poly + term


def polynomial_distance(poly1, poly2):
    distance = 0.0

    poly1_tiled = np.tile(poly1[:, np.newaxis, :], (poly2.shape[0], 1))
    difference = np.abs(poly1_tiled - poly2)
    distances = np.sum(difference, axis=2)
    min_distances_12 = np.min(distances, axis=0)
    sum_min_distances_12 = np.sum(min_distances_12)
    min_distances_21 = np.min(distances, axis=1)
    sum_min_distances_21 = np.sum(min_distances_21)

    distance += sum_min_distances_12 + sum_min_distances_21

    size1, size2 = poly1.shape[0], poly2.shape[0]

    distance /= (size1 + size2)

    return distance


def distances_norm(distances):
    return np.array([1]) / (np.array([1]) + distances)


def syn_coverage_norm(test_polynomial, polynomial_set):
    term_set = np.array(polynomial_set[0], dtype=np.int16)
    for polynomial in polynomial_set[1:]:
        term_set = np.vstack((term_set, polynomial))
    term_set_col = term_set[:, np.newaxis]
    difference = np.abs(test_polynomial - term_set_col)
    distances = np.sum(difference, axis=2)
    min_distances = np.min(distances, axis=0)
    max_similarities = distances_norm(min_distances)
    coverage = np.sum(max_similarities)
    return coverage


def syn_coverage_similarity(test_polynomial, polynomial_set):
    term_set = np.array(polynomial_set[0], dtype=np.int16)
    for polynomial in polynomial_set[1:]:
        term_set = np.vstack((term_set, polynomial))
    test_polynomial_norm = np.linalg.norm(test_polynomial, axis=1, keepdims=True)
    term_set_norm = np.linalg.norm(term_set, axis=1)[:, np.newaxis]
    dot_products = test_polynomial.dot(term_set.T)
    similarities = dot_products / (test_polynomial_norm.dot(term_set_norm.T))
    max_similarities = np.max(similarities, axis=1)
    coverage = np.sum(max_similarities)
    return coverage


def syn_set_coverage(test_polynomial, polynomial_set, mode):
    if mode == 'norm':
        return syn_coverage_norm(test_polynomial, polynomial_set) / len(test_polynomial)
    elif mode == 'similarity':
        return syn_coverage_similarity(test_polynomial, polynomial_set) / len(test_polynomial)
    

def word_coverage_overlap(test_tokenized, tokenized_trains):
    test_set = set(test_tokenized)
    train_set = set()
    for tokenized_train in tokenized_trains:
        train_set.update(tokenized_train)
    overlap = test_set.intersection(train_set)
    coverage = len(overlap) / len(test_set)
    return coverage
    

def word_set_coverage(test_tokenized, tokenized_trains):
    return word_coverage_overlap(test_tokenized, tokenized_trains)


def coverage(test_polynomial, test_tokenized, train_polynomials, train_tokenized, max_n, schema, syn_cov_mode, word_cov_weight):
    if schema == 'comb':
        alles_indices = np.array(range(len(train_polynomials)), dtype=np.int32)
        final_indices = np.array([], dtype=np.int32)
        current_indices = np.array([], dtype=np.int32)
        current_coverage = float('-inf')
        while len(final_indices) < max_n:
            next_idx, next_coverage = -1, float('-inf')
            for valid_idx in alles_indices:
                if valid_idx in final_indices:
                    continue
                tmp_indices = np.append(current_indices, valid_idx)
                tmp_syn_coverage = syn_set_coverage(test_polynomial, [train_polynomials[idx] for idx in tmp_indices], syn_cov_mode)
                tmp_word_coverage = word_set_coverage(test_tokenized, [train_tokenized[idx] for idx in tmp_indices])
                tmp_coverage = word_cov_weight * tmp_word_coverage + (1 - word_cov_weight) * tmp_syn_coverage
                if tmp_coverage > next_coverage:
                    next_idx = valid_idx
                    next_coverage = tmp_coverage
            if next_coverage > current_coverage:
                final_indices = np.append(final_indices, next_idx)
                current_indices = np.append(current_indices, next_idx)
                current_coverage = next_coverage
            else:
                current_indices = np.array([], dtype=np.int32)
                current_coverage = float('-inf')
        
        return final_indices
    
    elif schema == 'alternate-r':
        alles_indices = np.array(range(len(train_polynomials)), dtype=np.int32)
        final_indices = np.array([], dtype=np.int32)
        current_indices = np.array([], dtype=np.int32)
        current_syn_coverage = float('-inf')
        current_word_coverage = float('-inf')
        while len(final_indices) < max_n:
            next_idx, next_coverage = -1, float('-inf')
            for valid_idx in alles_indices:
                if valid_idx in final_indices:
                    continue
                tmp_indices = np.append(current_indices, valid_idx)
                if len(final_indices) % 2 == 1:
                    tmp_syn_coverage = syn_set_coverage(test_polynomial, [train_polynomials[idx] for idx in tmp_indices], syn_cov_mode)
                    # print(tmp_syn_coverage)
                    if tmp_syn_coverage > current_syn_coverage:
                        next_idx = valid_idx
                        current_syn_coverage = tmp_syn_coverage
                else:
                    tmp_word_coverage = word_set_coverage(test_tokenized, [train_tokenized[idx] for idx in tmp_indices])
                    if tmp_word_coverage > current_word_coverage:
                        next_idx = valid_idx
                        current_word_coverage = tmp_word_coverage
            if next_idx != -1:
                final_indices = np.append(final_indices, next_idx)
                current_indices = np.append(current_indices, next_idx)
            else:
                current_indices = np.array([], dtype=np.int32)
                if len(final_indices) % 2 == 1:
                    current_syn_coverage = float('-inf')
                else:
                    current_word_coverage = float('-inf')

        return final_indices

    elif schema == 'alternate':
        alles_indices = np.array(range(len(train_polynomials)), dtype=np.int32)
        final_indices = np.array([], dtype=np.int32)
        current_indices = np.array([], dtype=np.int32)
        current_syn_coverage = float('-inf')
        current_word_coverage = float('-inf')
        while len(final_indices) < max_n:
            next_idx, next_coverage = -1, float('-inf')
            for valid_idx in alles_indices:
                if valid_idx in final_indices:
                    continue
                tmp_indices = np.append(current_indices, valid_idx)
                if len(final_indices) % 2 == 0:
                    tmp_syn_coverage = syn_set_coverage(test_polynomial, [train_polynomials[idx] for idx in tmp_indices], syn_cov_mode)
                    if tmp_syn_coverage > current_syn_coverage:
                        next_idx = valid_idx
                        current_syn_coverage = tmp_syn_coverage
                else:
                    tmp_word_coverage = word_set_coverage(test_tokenized, [train_tokenized[idx] for idx in tmp_indices])
                    if tmp_word_coverage > current_word_coverage:
                        next_idx = valid_idx
                        current_word_coverage = tmp_word_coverage
            if next_idx != -1:
                final_indices = np.append(final_indices, next_idx)
                current_indices = np.append(current_indices, next_idx)
            else:
                current_indices = np.array([], dtype=np.int32)
                if len(final_indices) % 2 == 0:
                    current_syn_coverage = float('-inf')
                else:
                    current_word_coverage = float('-inf')

        return final_indices


def is_leaf(node):
    return len(node.children) == 0


def x2y(x, n_dim):
    return x + n_dim // 2


def ndim2nlabels(n_dim):
    return n_dim


def selection(lang, direction, split, cov_schema='comb', syn_cov_mode='norm', word_cov_weight=1.0, n=100, pre_selection=None, n_pre_selection=None):
    start_time = time.time()

    if direction == 'into':
        src_lang = lang
    else:
        src_lang = 'en'

    n_dim = lang2nlabels[src_lang]

    test_mfn = f'../data/{lang}/{split}.{src_lang}.moses'
    test_tfn = f'../data/{lang}/{split}.{src_lang}.spacy.json'
    train_mfn = f'../data/{lang}/train.{src_lang}.moses'
    train_tfn = f'../data/{lang}/train.{src_lang}.spacy.json'
    output_ifn = f'../data/{lang}/index/{split}/{direction}/coverage-{cov_schema}-{syn_cov_mode}-{word_cov_weight}.index'

    with open(output_ifn, 'w') as f:
        f.write('')
    f.close()

    print('=' * 60)
    print(f'{lang} {direction} {cov_schema} {syn_cov_mode} {word_cov_weight} {n}')

    print('Reading test tree file...')
    test_polynomials = read_tree_file(test_tfn, n_dim)
    test_tokenized = read_moses_file(test_mfn)
    assert len(test_polynomials) == len(test_tokenized)

    print('Reading train tree file...')
    train_polynomials = read_tree_file(train_tfn, n_dim)
    train_tokenized = read_moses_file(train_mfn)
    assert len(train_polynomials) == len(train_tokenized)

    pre_selection_idxs = None
    if pre_selection is not None:
        idx_ifn = f'../data/{lang}/index/{split}/{direction}/{pre_selection}.index'
        pre_selection_idxs = []
        with open(idx_ifn, 'r') as f:
            for line in f:
                line = line.strip().split()
                idxs = [int(idx) for idx in line[:n_pre_selection]]
                if len(idxs) < n_pre_selection:
                    all_idxs = list(range(len(train_polynomials)))
                    sample_pool = list(set(all_idxs) - set(idxs))
                    sample_idxs = random.sample(sample_pool, n_pre_selection - len(idxs))
                    idxs += sample_idxs
                pre_selection_idxs.append(idxs)
        pre_selection_idxs = np.array(pre_selection_idxs, dtype=np.int32)

    if pre_selection is not None:
        pool_size = n_pre_selection
    else:
        pool_size = len(train_polynomials)

    print('Calculating distances...')

    for i, test_polynomial in enumerate(tqdm(test_polynomials, ncols=60)):
        test_tokenized_sentence = test_tokenized[i]
        current_train_polynomials = train_polynomials
        current_train_tokenized = train_tokenized
        if pre_selection is not None:
            current_train_polynomials = [train_polynomials[idx] for idx in pre_selection_idxs[i][:pool_size]]
            current_train_tokenized = [train_tokenized[idx] for idx in pre_selection_idxs[i][:pool_size]]
        
        top_n_idxs = coverage(test_polynomial, test_tokenized_sentence, current_train_polynomials, current_train_tokenized, n, cov_schema, syn_cov_mode, word_cov_weight)

        if pre_selection is not None:
            top_n_idxs = pre_selection_idxs[i][top_n_idxs]
        
        with open(output_ifn, 'a') as f:
            f.write(' '.join([str(idx) for idx in top_n_idxs]) + '\n')
        f.close()
        
    print(f'Elapsed time: {time.time() - start_time:.2f}s')


def main():
    langs = ['de', 'fr', 'ru']
    directions = ['into', 'outof']
    split = 'test'

    for lang in langs:
        for direction in directions:
            # SCOI
            selection(lang, direction, split=split, cov_schema='alternate', syn_cov_mode='norm', word_cov_weight=0, n=8, pre_selection='bm25', n_pre_selection=100)
            
            # w/o syntax variant
            selection(lang, direction, split=split, cov_schema='comb', syn_cov_mode='norm', word_cov_weight=1, n=8, pre_selection='bm25', n_pre_selection=100)

            # w/o word variant
            selection(lang, direction, split=split, cov_schema='comb', syn_cov_mode='norm', word_cov_weight=0, n=8, pre_selection='bm25', n_pre_selection=100)

            # word-first variant
            selection(lang, direction, split=split, cov_schema='alternate-r', syn_cov_mode='norm', word_cov_weight=0, n=8, pre_selection='bm25', n_pre_selection=100)

            # cosine similarity variant
            selection(lang, direction, split=split, cov_schema='alternate-r', syn_cov_mode='similarity', word_cov_weight=0, n=8, pre_selection='bm25', n_pre_selection=100)


if __name__ == '__main__':
    main()