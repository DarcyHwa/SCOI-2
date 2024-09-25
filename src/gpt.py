import random
import os

from tqdm import tqdm
from openai import OpenAI


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


def get_prompt(test_sentence, train_pairs, src_lang, tgt_lang, template="gpt"):
    if template == "gpt":
        return template_gpt(test_sentence, train_pairs, src_lang, tgt_lang)
    

def template_gpt(test_sentence, train_pairs, src_lang, tgt_lang):
    src_lang = lang_map(src_lang)
    tgt_lang = lang_map(tgt_lang)
    prompt = []
    system_instructions = {
        'role': 'system',
        'content': f'You are a helpful translator. The user will give you {src_lang} sentences, and you need to translate them into {tgt_lang}.'
    }
    prompt.append(system_instructions)

    for train_pair in train_pairs:
        user_message = {
            'role': 'user',
            'content': f'{train_pair[0]}'
        }
        assistant_message = {
            'role': 'assistant',
            'content': f'{train_pair[1]}'
        }
        prompt.append(user_message)
        prompt.append(assistant_message)
    
    user_message = {
        'role': 'user',
        'content': f'{test_sentence}'
    }
    prompt.append(user_message)

    return prompt


def extract_answer(generated_text, template="gpt"):
    if template == "gpt":
        return extract_answer_gpt(generated_text)
    

def extract_answer_gpt(generated_text):
    generated_text = generated_text.strip()
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


def main(selections=["bm25-polynomial"], order="descending", langs=["de", "fr", "ru"], directions=["into", "outof"], output_dir="../output/gpt", shot=4, template="gpt", cut=-1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    client = OpenAI(
        base_url = "https://api.openai-proxy.org/v1",
        api_key="sk-", # Add your API key here
    )

    for direction in directions:
        for lang in langs:
            for selection in selections:
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
                idx_fn = f"{idx_dir}/{selection_list[0]}.index"
                idx_list = read_idx_file(idx_fn)

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
                        idxs = idx_list[i]
                        train_pairs = idx2example(train_sentence_pairs, idxs, shot, order)
                        prompt = get_prompt(test_sentence, train_pairs, src_lang, tgt_lang, template=template)
                        prompts.append(prompt)

                    for prompt in tqdm(prompts, ncols=60):
                        response = client.chat.completions.create(
                            messages=prompt,
                            model="gpt-3.5-turbo-0125",
                            max_tokens=256,
                        )

                        output = response.choices[0].message.content
                        output = extract_answer(output, template)
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
    selections = ["rand3"]
    order = "ascending"
    langs = ["de", "fr", "ru"]
    directions = ["into", "outof"]
    output_dir = "../output/gpt"
    shot = 4
    template = "gpt"
    cut = -1
    main(selections, order, langs, directions, output_dir, shot, template, cut)