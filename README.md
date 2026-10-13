# GPT-2 (124M) from scratch

A PyTorch reimplementation of the GPT-2 124M decoder-only Transformer, pretrained on a ~500M-token
slice of FineWeb-Edu and evaluated on HellaSwag. The code closely follows Andrej Karpathy's
[build-nanoGPT](https://github.com/karpathy/build-nanoGPT) ("Let's reproduce GPT-2 (124M)"), with a
smaller dataset pipeline and Weights & Biases logging added.

## What's here

`train_gpt2.py` contains the model and the training loop in one file:

- `CausalSelfAttention` - fused QKV projection, causal attention via `F.scaled_dot_product_attention`
  (uses FlashAttention kernels where PyTorch supports them).
- `MLP` - `Linear -> GELU (tanh approx.) -> Linear`, 4x expansion.
- `Block` - pre-LayerNorm residual block (`x + attn(ln_1(x))`, `x + mlp(ln_2(x))`).
- `GPT` - token + learned position embeddings, 12 blocks, final LayerNorm, LM head tied to the token
  embedding. Residual projections use GPT-2's scaled init (`std * (2 * n_layer) ** -0.5`).
  `GPT.from_pretrained("gpt2")` loads OpenAI's weights from Hugging Face for comparison.
- `GPT.configure_optimizers` - AdamW (`betas=(0.9, 0.95)`, weight decay 0.1 on 2-D params only, fused on CUDA).
- `DataLoaderLite` - streams contiguous `B x T` token windows from `.npy` shards.
- Training loop - linear warmup + cosine LR decay (`max_lr=6e-4`), gradient clipping at 1.0, bf16
  autocast, DDP support via `torchrun`; every 250 steps it runs validation loss, HellaSwag, and
  sample generation, writing to `log/log.txt` and W&B.

Model config (`GPTConfig`): `n_layer=12`, `n_head=12`, `n_embd=768`, `block_size=1024`; the vocab is
padded from 50257 to 50304 for better GPU efficiency.

Supporting files:

- `fineweb.py` - streams `HuggingFaceFW/fineweb-edu` (`sample-10BT`), tokenizes with the GPT-2
  `tiktoken` encoding, and writes 5 shards of 100M `uint16` tokens to `edu_fineweb100M/`
  (shard 0 = val, shards 1-4 = train).
- `hellaswag.py` - renders HellaSwag examples into 4 candidate completions and scores them by average
  completion loss. Also runnable on its own to evaluate pretrained Hugging Face GPT-2 checkpoints.
- `hellaswag/hellaswag_val.jsonl` - cached HellaSwag validation set (downloaded automatically if missing).
- `Play.ipynb` - exploration notebook: inspects the Hugging Face GPT-2 state dict, plots position
  embeddings, samples from the pretrained model, and tokenizes Tiny Shakespeare (`input.txt`).
- `log/log.txt`, `steps-data.txt` - log and console output from a training run (see below).

## Logged run

`log/log.txt` records a single-GPU run that stopped after ~1,000 steps:

| Step | Val loss | HellaSwag acc_norm |
| --- | --- | --- |
| 0 | 10.9511 | 0.2491 |
| 500 | 6.5821 | 0.2443 |
| 1000 | 6.3299 | 0.2421 |

HellaSwag is still at chance level (0.25) at this point. Note that the training loop computes
`grad_accum_steps` but does not accumulate gradients, so each optimizer step uses a single micro-batch
of `8 x 1024` tokens rather than the intended ~0.5M-token batch.

## Getting started

```bash
git clone https://github.com/khushwant04/gpt-2.git
cd gpt-2
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # install a CUDA build of PyTorch first if needed

# 1. Build the token shards (~500M tokens, about 1 GB on disk)
python fineweb.py

# 2. Train (single GPU)
python train_gpt2.py

# or with DDP across N GPUs
torchrun --standalone --nproc_per_node=N train_gpt2.py
```

`train_gpt2.py` calls `wandb.init(project="gpt2")` at import time. Run `wandb login` first, or
disable it with `WANDB_MODE=disabled python train_gpt2.py`.

Checkpoints are written to `log/model_<step>.pt` every 5,000 steps and at the last step.

Evaluate pretrained OpenAI GPT-2 on HellaSwag (via Hugging Face `transformers`):

```bash
python hellaswag.py -m gpt2 -d cuda      # also gpt2-medium, gpt2-large, gpt2-xl
```

## Project structure

```
.
├── train_gpt2.py          # model + training loop
├── fineweb.py             # dataset download + tokenization into shards
├── hellaswag.py           # HellaSwag rendering and evaluation
├── hellaswag/             # cached HellaSwag val split
├── Play.ipynb             # exploration notebook
├── input.txt              # Tiny Shakespeare (used by the notebook)
├── log/log.txt            # step / val / hella / train log
└── steps-data.txt         # console output of the logged run
```

## References

- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2), 2019. [PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- Brown et al., *Language Models are Few-Shot Learners* (GPT-3; source of the optimizer and LR-schedule hyperparameters), 2020. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
- Vaswani et al., *Attention Is All You Need*, 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Dao et al., *FlashAttention*, 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Zellers et al., *HellaSwag: Can a Machine Really Finish Your Sentence?*, 2019. [arXiv:1905.07830](https://arxiv.org/abs/1905.07830)
- Penedo et al., *The FineWeb Datasets*, 2024. [arXiv:2406.17557](https://arxiv.org/abs/2406.17557)
- Andrej Karpathy, [build-nanoGPT](https://github.com/karpathy/build-nanoGPT) and [nanoGPT](https://github.com/karpathy/nanoGPT).
