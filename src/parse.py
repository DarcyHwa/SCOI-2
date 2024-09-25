import json
import spacy

from time import time
from tqdm import tqdm


lang_to_path = {
    'en': "en_core_web_sm",
    'de': "de_core_news_sm",
    'fr': 'fr_core_news_sm',
    'ru': 'ru_core_news_sm',
}


def main(lang):
    nlps = {}

    nlps[lang] = spacy.load(lang_to_path[lang])
    nlps['en'] = spacy.load(lang_to_path['en'])

    dep2idx_en = depidx('en')
    dep2idx_lang = depidx(lang)

    parse_test_data(lang, nlps, dep2idx_en, dep2idx_lang)
    parse_train_data(lang, nlps, dep2idx_en, dep2idx_lang)


def depidx(lang):
    dep2idx = {}
    fn = f'../data/dependency/{lang}.txt'
    with open(fn, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if len(line.strip()) == 0:
                continue
            dep2idx[line.strip()] = idx
    return dep2idx


def parse_train_data(lang, nlps, dep2idx_en, dep2idx_de):
    en_fn = f'../data/{lang}/train.en'
    # de refers to the second language
    de_fn = f'../data/{lang}/train.{lang}'

    in_en = open(en_fn, 'r', encoding='utf-8')
    in_de = open(de_fn, 'r', encoding='utf-8')
    out_en = open(en_fn + '.spacy.json', 'w', encoding='utf-8')
    out_de = open(de_fn + '.spacy.json', 'w', encoding='utf-8')

    ens = []
    des = []

    start = time()
    for line in tqdm(in_en):
        ens.append(line.strip())
    
    for line in tqdm(in_de):
        des.append(line.strip())

    assert len(ens) == len(des)

    out_en_list = []
    for idx, en_doc in enumerate(tqdm(nlps['en'].pipe(ens, batch_size=1024), ncols=60, total=len(ens))):
        tree = []
        for token in en_doc:
            token_i = token.i + 1
            head_i = token.head.i + 1 if token.dep_ != 'ROOT' else 0
            dep_i = dep2idx_en[token.dep_]
            tree.append((token_i, head_i, dep_i))
        out_en_list.append({'id': idx, 'tree': tree})
    
    json.dump(out_en_list, out_en, ensure_ascii=False, indent=4)
    out_en.write('\n')
    
    out_de_list = []
    for idx, de_doc in enumerate(tqdm(nlps[lang].pipe(des, batch_size=1024), ncols=60, total=len(des))):
        tree = []
        for token in de_doc:
            token_i = token.i + 1
            head_i = token.head.i + 1 if token.dep_ != 'ROOT' else 0
            dep_i = dep2idx_de[token.dep_]
            tree.append((token_i, head_i, dep_i))
        out_de_list.append({'id': idx, 'tree': tree})
    
    json.dump(out_de_list, out_de, ensure_ascii=False, indent=4)
    out_de.write('\n')

    print(f"Finished processing train.en and train.{lang} in {time() - start} seconds")

    # close all files
    in_en.close()
    in_de.close()
    out_en.close()
    out_de.close()


def parse_test_data(lang, nlps, dep2idx_en, dep2idx_de):
    en_fn = f'../data/{lang}/test.en'
    de_fn = f'../data/{lang}/test.{lang}'

    in_en = open(en_fn, 'r', encoding='utf-8')
    in_de = open(de_fn, 'r', encoding='utf-8')
    out_en = open(en_fn + '.spacy.json', 'w', encoding='utf-8')
    out_de = open(de_fn + '.spacy.json', 'w', encoding='utf-8')

    ens = []
    des = []

    start = time()  
    for line in tqdm(in_en):
        ens.append(line.strip())

    for line in tqdm(in_de):
        des.append(line.strip())

    assert len(ens) == len(des)

    out_en_list = []
    for idx, en_doc in enumerate(tqdm(nlps['en'].pipe(ens, batch_size=1024), ncols=60, total=len(ens))):
        tree = []
        for token in en_doc:
            token_i = token.i + 1
            head_i = token.head.i + 1 if token.dep_ != 'ROOT' else 0
            dep_i = dep2idx_en[token.dep_]
            tree.append((token_i, head_i, dep_i))
        out_en_list.append({'id': idx, 'tree': tree})

    json.dump(out_en_list, out_en, ensure_ascii=False, indent=4)
    out_en.write('\n')
    
    out_de_list = []
    for idx, de_doc in enumerate(tqdm(nlps[lang].pipe(des, batch_size=1024), ncols=60, total=len(des))):
        tree = []
        for token in de_doc:
            token_i = token.i + 1
            head_i = token.head.i + 1 if token.dep_ != 'ROOT' else 0
            dep_i = dep2idx_de[token.dep_]
            tree.append((token_i, head_i, dep_i))
        out_de_list.append({'id': idx, 'tree': tree})

    json.dump(out_de_list, out_de, ensure_ascii=False, indent=4)
    out_de.write('\n')

    print(f"Finished processing test.en and test.{lang} in {time() - start} seconds")

    # close all files
    in_en.close()
    in_de.close()
    out_en.close()
    out_de.close()


if __name__ == "__main__":
    langs = ['de', 'fr', 'ru']
    for lang in langs:
        main(lang)