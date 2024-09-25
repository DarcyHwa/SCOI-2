import random
import os

from tqdm import tqdm

langs = ['de', 'fr', 'ru']
directions = ['into', 'outof']

def get_train_size(lang):
    train_fn = f'../data/{lang}/train.{lang}'
    with open(train_fn, 'r') as f:
        return len(f.readlines())
    
def get_test_size(lang):
    test_fn = f'../data/{lang}/test.{lang}'
    with open(test_fn, 'r') as f:
        return len(f.readlines())
    
def get_test_size(lang):
    test_fn = f'../data/{lang}/test.{lang}'
    with open(test_fn, 'r') as f:
        return len(f.readlines())

def alles_rand():
    for _ in [1,2,3]:
        random.seed(_)
        for lang in langs:
            for direction in directions:
                output_fn = f'../data/{lang}/index/test/{direction}/rand{_}.index'
                if not os.path.exists(f'../data/{lang}/index/test/{direction}'):
                    os.makedirs(f'../data/{lang}/index/test/{direction}')
                train_size, test_size = get_train_size(lang), get_test_size(lang)
                with open(output_fn, 'w') as f:
                    for i in tqdm(range(test_size)):
                        # randomly select k training examples
                        indexs = random.sample(range(train_size), 16)
                        indexs = [str(index) for index in indexs]
                        f.write(' '.join(indexs) + '\n')

alles_rand()