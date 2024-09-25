import random
import torch
import os

from tqdm import tqdm
from transformers import XGLMTokenizer, XGLMForCausalLM, pipeline
from torch.utils.data import Dataset


class MTDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, i):
        return self.prompts[i]


def lang_map(lang):
    lang_dict = {'en': 'English', 'de': 'German', 'fr': 'French', 'ru': 'Russian'}
    return lang_dict[lang]


def idx2example(train_sentence_pairs, idxs, shot, order):
    THRESHOLD = 120
    final_idxs = []
    final_examples = []
    count, i = 0, 0

    while count < shot and i < len(idxs):
        idx = idxs[i]
        src = train_sentence_pairs[idx][0].strip('"').split()
        if len(src) > THRESHOLD:
            i += 1
            continue
        final_idxs.append(idx)
        i += 1
        count += 1
    
    while count < shot:
        idx = random.randint(0, len(train_sentence_pairs) - 1)
        src = train_sentence_pairs[idx][0].strip('"').split()
        if len(src) > THRESHOLD:
            continue
        final_idxs.append(idx)
        count += 1

    for idx in final_idxs:
        final_examples.append(train_sentence_pairs[idx])

    final_examples = do_order(final_examples, order)

    return final_examples


def get_prompt(test_sentence, train_pairs_list, src_lang, tgt_lang, template="a"):
    if template == "xglm":
        return template_xglm(test_sentence, train_pairs_list, src_lang, tgt_lang)
    else:
        raise ValueError("Invalid template")


def template_xglm(test_sentence, train_pairs_list, src_lang, tgt_lang):
    # From CTQScorer (Kumar et al., 2023)
    src_lang = lang_map(src_lang)
    tgt_lang = lang_map(tgt_lang)
    prompt = []
    for train_pair in train_pairs_list[0]:
        content = """{} Sentence: "{}"
{} Sentence: "{}"
###
""".format(src_lang, train_pair[0], tgt_lang, train_pair[1])
        prompt.append(content)
    test_content = """{} Sentence: "{}"
{} Sentence: """.format(src_lang, test_sentence, tgt_lang)
    prompt.append(test_content)
    prompt = "".join(prompt)
    return prompt


def extract_answer(generated_text):
    generated_text = generated_text.strip()
    generated_text = generated_text.split('###')[0]
    generated_text = generated_text.strip().replace('\n', '')
    return generated_text


def read_idx_file(fn):
    idx_list = []
    with open(fn, "r") as f:
        for line in f:
            line = line.strip()
            idxs = line.split(" ")
            idxs = [int(idx) for idx in idxs]
            idx_list.append(idxs)
    return idx_list


def do_order(list, order):
    if order == "descending":
        return list
    elif order == "ascending":
        return list[::-1]
    elif order == "random":
        return random.shuffle(list)


def main(device="cuda:0", selections=["bm25"], order="descending", langs=["de", "fr", "ru"], directions=["into", "outof"], model_path="facebook/xglm-7.5B", output_dir="../output/xglm", shot=4, batch_size=8, template="xglm", cut=-1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    torch.device(device)
    tokenizer = XGLMTokenizer.from_pretrained(model_path, padding_side="left")
    model = XGLMForCausalLM.from_pretrained(model_path).to(device)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False, device=device)

    for selection in selections:
        for direction in directions:
            for lang in langs:
                if direction == "into":
                    src_lang = lang
                    tgt_lang = "en"
                    test_src_fn = f"../data/{lang}/test.{lang}"
                    test_tgt_fn = f"../data/{lang}/test.en"
                    train_src_fn = f"../data/{lang}/train.{lang}"
                    train_tgt_fn = f"../data/{lang}/train.en"
                    idx_dir = f"../data/{lang}/index/test/into"
                else:
                    src_lang = "en"
                    tgt_lang = lang
                    test_src_fn = f"../data/{lang}/test.en"
                    test_tgt_fn = f"../data/{lang}/test.{lang}"
                    train_src_fn = f"../data/{lang}/train.en"
                    train_tgt_fn = f"../data/{lang}/train.{lang}"
                    idx_dir = f"../data/{lang}/index/test/outof"

                selection_list = selection.split("+")
                idx_list_list = []
                for sel in selection_list:
                    idx_fn = f"{idx_dir}/{sel}.index"
                    idx_list = read_idx_file(idx_fn)
                    idx_list_list.append(idx_list)

                test_sentences = []
                with open(test_src_fn, "r") as f:
                    for line in f:
                        test_sentences.append(line.strip())
                gold = []
                with open(test_tgt_fn, "r") as f:
                    for line in f:
                        gold.append(line.strip())
                if cut > 0:
                    test_sentences = test_sentences[:cut]
                    gold = gold[:cut]
                
                train_sentence_pairs = []
                with open(train_src_fn, "r") as f1, open(train_tgt_fn, "r") as f2:
                    for src, tgt in zip(f1, f2):
                        train_sentence_pairs.append((src.strip(), tgt.strip()))

                output_fn = f"{output_dir}/{lang}.{direction}.{selection}.{shot}.{order}.{template}.txt"

                system = []
                with open(output_fn, "w") as f:
                    prompts = []
                    for i in tqdm(range(len(test_sentences)), ncols=60):
                        test_sentence = test_sentences[i]
                        train_pairs_list = []
                        count_dict = {}
                        for sel in selection_list:
                            count_dict[sel] = 0
                        for j in range(len(selection_list)):
                            idx_list = idx_list_list[j]
                            sel = selection_list[j]
                            train_pairs = idx2example(train_sentence_pairs, idx_list[i][count_dict[sel]*(shot//len(idx_list_list)):], shot//len(idx_list_list), order)
                            train_pairs_list.append(train_pairs)
                            count_dict[sel] += 1
                    
                        prompt = get_prompt(test_sentence, train_pairs_list, src_lang, tgt_lang, template)
                        prompts.append(prompt)

                    dataset = MTDataset(prompts)

                    for out in tqdm(pipe(dataset, max_new_tokens=128, batch_size=batch_size, do_sample=False), ncols=60, total=len(prompts)):
                        output = out[0]['generated_text']
                        output = extract_answer(output)
                        system.append(output)
                        f.write(output + "\n")
                
                print("=====================================")
                print(f"Language: {lang}")
                print(f"Direction: {direction}")
                print(f"Selection: {selection}")
                print(f"Shot: {shot}")
                print(f"Order: {order}")
                print(f"Template: {template}")
                print("=====================================")


if __name__ == "__main__":
    device = "cuda"
    selections = ["rand1", "rand2", "rand3", "bm25", "rbm25", "fuzzy", "coverage-alternate-norm-0", "ctq"]
    order = "ascending"
    langs = ["de", "fr", "ru"]
    directions = ["into", "outof"]
    model_path = "facebook/xglm-7.5B"
    output_dir = "../output/xglm"
    shot = 4
    batch_size = 2
    template = "xglm"
    cut = -1
    main(device, selections, order, langs, directions, model_path, output_dir, shot, batch_size, template, cut)