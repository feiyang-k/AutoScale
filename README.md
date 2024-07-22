#### Anonymous Code Repository for 
### [AutoScale–Automatic Prediction of Compute-optimal Data Compositions for Training LLMs]

![Python 3.9.10](https://img.shields.io/badge/python-3.9.10-DodgerBlue.svg?style=plastic)

**Abstract:** To ensure performance on a diverse set of downstream tasks, LLMs are pretrained via data mixtures over different domains. In this work, we demonstrate that the optimal data composition for a fixed compute budget varies depending on the scale of the training data, suggesting that the common practice of empirically determining an optimal composition using small-scale experiments will not yield the optimal data mixtures when scaling up to the final model. To address this challenge, we propose AutoScale, an automated tool that finds a compute-optimal data composition for training at any desired target scale. AutoScale first determines the optimal composition at a small scale using a novel bilevel optimization framework, {D}irect {D}ata {O}ptimization (DDO), and then fits a predictor to estimate the optimal composition at larger scales. The predictor's design is inspired by our theoretical analysis of scaling laws related to data composition, which could be of independent interest. In empirical studies with pre-training 774M Decoder-only LMs (GPT-2 Large) on RedPajama dataset, AutoScale decreases validation perplexity at least 25% faster than any baseline with up to 38% speed up compared to without reweighting, achieving the best overall performance across downstream tasks. On pre-training Encoder-only LMs (BERT) with masked language modeling, DDO is shown to decrease loss on all domains while visibly improving average task performance on GLUE benchmark by 8.7% and on large-scale QA dataset (SQuAD) by 5.9% compared with without reweighting. AutoScale speeds up training by up to 28%. Our codes are open-sourced.

**For experiments on Decoder-only LMs (GPT-2 Large):** Run
1. prepare_raw_data.py
2. prepare_data.py
3. run_gpt2_large.sh
   
**For experiments on Encoder-only LMs (BERT-base):** Run
- Interactive example: simple_bert.ipynb
