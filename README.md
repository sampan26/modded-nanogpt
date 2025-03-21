# modded-nanogpt

Inspired by Modded NanoGPT by Keller Jordan, this variant comes from the code provided by Kaparthy on his Lets Build GPT-2 (124M) video. It
* Trains 2x more efficiently (taking only 5B tokens instead of 10B to reach the same validation loss).
* Implements modernizations like rotary embeddings


To run it:
```
python data/fineweb.py
torchrun --n_proc_per_node=# train_gpt.py
```

This will produce a 124M-parameter transformer trained on 6.4B tokens (139.3 minutes), which has 3.125 validation loss on the Fineweb validation set. For comparison, the original trainer yields 3.2847 validation loss after training for 10B tokens (260.17 minutes)

The speedup is due to the following changes:
- Increased learning rate by 3x
- Switched to rotary embeddings
- Implemented RMSNorm (slightly faster than LayerNorm and Pytorch RMSNorm for some reason)


Baseline loss: 3.126; Training Time: 260.17 minutes; avg step: 397ms
3/19 loss: 3.125; Training Time: 139.93 minutes, avg step: 330ms
