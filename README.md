# Mastering Diverse Domains through World Models

A reimplementation of [DreamerV3][paper] with bisimulation metrics based on the paper [Learning Invariant Representations for Reinforcement Learning without Reconstruction](https://arxiv.org/abs/2006.10742)

# Instructions

The code has been tested on Linux and requires Python 3.11+.


Install [JAX][jax] and then the other dependencies:

```sh
pip install jax==0.5.0
#or if you have cuda
pip install jax[cuda]==0.5.0
pip install -U -r requirements.txt
```

Training script:

```sh
python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/{timestamp} \
  --configs dmc_finger\
  --run.train_ratio 32
```

To look how the model trains you can run plot.py

Scalar metrics are also writting as JSONL files.

# Tips you can look [here](https://github.com/facebookresearch/deep_bisim4control/tree/main)



