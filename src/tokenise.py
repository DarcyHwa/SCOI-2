from sacremoses import MosesTokenizer, MosesPunctNormalizer
from tqdm import tqdm

langs = ['de', 'fr', 'ru']
dirs = ['into', 'outof']
splits = ['train', 'test']

for lang in langs:
    for d in dirs:
        for split in splits:
            print(f'Processing {lang} {d} {split}...')
            if d == 'into':
                src_lang = lang
            else:
                src_lang = 'en'
            text_fn = f'../data/{lang}/{split}.{src_lang}'
            token_fn = f'../data/{lang}/{split}.{src_lang}.moses'
            with open(text_fn, 'r') as f:
                text = f.readlines()
            mt = MosesTokenizer(lang=src_lang)
            mpn = MosesPunctNormalizer(lang=src_lang)
            with open(token_fn, 'w') as f:
                for line in tqdm(text):
                    line = line.strip()
                    line = mpn.normalize(line)
                    line = mt.tokenize(line, return_str=True)
                    f.write(line + '\n')