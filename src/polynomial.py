# Implementation of Polynomial from SCOI (Anonymous, 2024)

import json
import numpy as np
import random
import time

from multiprocessing import Pool
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

    n_labels = ndim2nlabels(n_dim)
    terms = 0
    
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


def is_leaf(node):
    return len(node.children) == 0


def x2y(x, n_dim):
    return x + n_dim // 2


def ndim2nlabels(n_dim):
    return n_dim


def worker(args):
    train_polynomial, test_polynomial = args
    score = polynomial_distance(test_polynomial, train_polynomial)
    return score


def selection(lang, direction, split, n=100, pre_selection=None, n_pre_selection=None):
    start_time = time.time()

    if direction == 'into':
        src_lang = lang
    else:
        src_lang = 'en'

    method = 'polynomial'

    n_dim = lang2nlabels[src_lang]

    test_tfn = f'../data/{lang}/{split}.{src_lang}.spacy.json'
    train_tfn = f'../data/{lang}/train.{src_lang}.spacy.json'
    output_ifn = f'../data/{lang}/index/{split}/{direction}/polynomial.index'
    output_sfn = f'../data/{lang}/index/{split}/{direction}/polynomial.score'
    print(f'output: {output_ifn}')

    with open(output_ifn, 'w') as f:
        f.write('')
    f.close()

    with open(output_sfn, 'w') as f:
        f.write('')
    f.close()

    print('=' * 60)
    print(f'{lang} {direction} {method}')

    print('Reading test tree file...')
    test_polynomials = read_tree_file(test_tfn, n_dim)

    print('Reading train tree file...')
    train_polynomials = read_tree_file(train_tfn, n_dim)

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
        scores = []

        current_train_polynomials = train_polynomials
        if pre_selection is not None:
            current_train_polynomials = [train_polynomials[idx] for idx in pre_selection_idxs[i][:pool_size]]
        
        current_test_polynomials = [test_polynomial] * len(current_train_polynomials)
        
        with Pool(4) as p:
            scores = p.map(worker, zip(current_train_polynomials, current_test_polynomials))
        p.close()
        
        scores = np.array(scores, dtype=np.float32)
        top_n_idxs = np.argsort(scores)[:n]
        top_n_scores = scores[top_n_idxs]

        if pre_selection is not None:
            top_n_idxs = pre_selection_idxs[i][top_n_idxs]
        
        with open(output_ifn, 'a') as f:
            f.write(' '.join([str(idx) for idx in top_n_idxs]) + '\n')
        f.close()
        
        with open(output_sfn, 'a') as f:
            f.write(' '.join([str(score) for score in top_n_scores]) + '\n')
        f.close()

    print(f'Elapsed time: {time.time() - start_time:.2f}s')


def main():
    langs = ['de', 'fr', 'ru']
    directions = ['into', 'outof']
    split = 'test'

    for lang in langs:
        for direction in directions:
            selection(lang, direction, split=split, n=100, pre_selection='bm25', n_pre_selection=100)


if __name__ == '__main__':
    main()