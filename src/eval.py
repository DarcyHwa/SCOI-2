import torch
import os

from score import init_comet_20, get_comet_20_mean_score, get_bleu_mean_score


def main(model="alpaca", device="cuda", selections=["bm25"], order="descending", langs=["de", "fr", "ru"], directions=["into", "outof"], shot=4, template="alpaca", log_path="../result/alpaca/result.tsv"):
    if not os.path.exists(f'../result/{model}'):
        os.makedirs(f'../result/{model}')
    
    torch.device(device)
    comet_20_metric = init_comet_20("Unbabel/wmt20-comet-da")

    for direction in directions:
        for lang in langs:
            for selection in selections:
                if direction == "into":
                    test_src_fn = f"../data/{lang}/test.{lang}"
                    test_tgt_fn = f"../data/{lang}/test.en"
                else:
                    test_src_fn = f"../data/{lang}/test.en"
                    test_tgt_fn = f"../data/{lang}/test.{lang}"

                test_sentences = []
                with open(test_src_fn, "r") as f:
                    for line in f:
                        test_sentences.append(line.strip())
                gold = []
                with open(test_tgt_fn, "r") as f:
                    for line in f:
                        gold.append(line.strip())
                
                output_fn = f"../output/{model}/{lang}.{direction}.{selection}.{shot}.{order}.{template}.txt"

                system = []
                with open(output_fn, "r") as f:
                    for line in f:
                        system.append(line.strip())
                
                source = test_sentences

                comet_20_score = get_comet_20_mean_score(system, gold, source, comet_20_metric)
                bleu_score = get_bleu_mean_score(system, gold)
                print("=====================================")
                print(f"Language: {lang}")
                print(f"Direction: {direction}")
                print(f"Selection: {selection}")
                print(f"Shot: {shot}")
                print(f"Order: {order}")
                print(f"Template: {template}")
                print('-------------------------------------')
                print(f"COMET-20: {100 * comet_20_score}")
                print(f"BLEU: {bleu_score}")
                print("=====================================")

                with open(log_path, 'a') as f_log:
                    f_log.write(f"{template}\t{shot}\t{lang}\t{direction}\t{selection}\t{order}\t{100 * comet_20_score}\t{bleu_score}\n")
                f_log.close()


if __name__ == "__main__":
    device = "cuda"
    selections = ["rand1", "rand2", "rand3", "bm25", "rbm25", "fuzzy", "coverage-alternate-norm-0", "ctq"]
    order = "ascending"
    langs = ["de", "fr", "ru"]
    directions = ["into", "outof"]
    shot = 4

    model = "xglm"
    template = "xglm"
    log_path = f"../result/{model}/result.tsv"
    main(model, device, selections, order, langs, directions, shot, template, log_path)

    model = "alpaca"
    template = "alpaca"
    log_path = f"../result/{model}/result.tsv"
    main(model, device, selections, order, langs, directions, shot, template, log_path)