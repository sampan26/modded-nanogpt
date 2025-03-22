# modded-nanogpt

Inspired by Modded NanoGPT by Keller Jordan, this variant comes from the code provided by Kaparthy on his Lets Build GPT-2 (124M) video. It
* Trains 5x more efficiently (taking only 2B tokens instead of 10B to reach the same validation loss).
* Implements modernizations like rotary embeddings and RMS norm

The runs were conducted on a 4xA100 cluster with speedups if trained with more gpus.

To run it:
```
python data/fineweb.py
torchrun --n_proc_per_node=# train_gpt.py
```

This will train a 124M-parameter transformer trained for 7250 steps on 1.8B tokens, which has 3.125 validation loss on the Fineweb validation set. For comparison, the original train_gpt.py yields 3.126 validation loss after training for 10B tokens for 260.17 minutes (on a 4xA100)

The speedup is due to the following changes:
- Increased learning rate by 3x
- Switched to rotary embeddings and ReLU^2 activation
- Implemented RMSNorm (slightly faster than LayerNorm and Pytorch RMSNorm for some reason)
- Switched from AdamW to SOAP optimizer
- Changed to 6 attention heads instead of 12



Baseline loss: 3.126; Training Time: 260.17 minutes; avg step: 397ms
3/19 loss: 3.125; Training Time: 139.93 minutes, avg step: 330ms
3/21 loss: 3.124; Training Time: 48.31 minutes, avg step: 362 ms
