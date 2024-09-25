export CUDA_VISIBLE_DEVICES=0

# Inference (XGLM)
python xglm.py

# Inference (Alpaca)
python alpaca.py

# Evaluation (COMET-20)
python eval.py
