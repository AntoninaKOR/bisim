# Mastering Diverse Domains through World Models

A reimplementation of [DreamerV3][paper] with bisimulation metrics based on the paper [Learning Invariant Representations for Reinforcement Learning without Reconstruction](https://arxiv.org/abs/2006.10742)

## DreamerV3

DreamerV3 learns a world model from experiences and uses it to train an actor
critic policy from imagined trajectories. The world model encodes sensory
inputs into categorical representations and predicts future representations and
rewards given actions.



# Instructions

The code has been tested on Linux and requires Python 3.11+.

## Docker

You can either use the provided `Dockerfile` that contains instructions or
follow the manual instructions below.

## Manual

Install [JAX][jax] and then the other dependencies:

```sh
pip install -U -r requirements.txt
```

Training script:

```sh
python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/{timestamp} \
  --configs crafter \
  --run.train_ratio 32
```

To reproduce results, train on the desired task using the corresponding config,
such as `--configs atari --task atari_pong`.

To look how the model trains you can run plot.py

Scalar metrics are also writting as JSONL files.

# Tips you can look [here](https://github.com/facebookresearch/deep_bisim4control/tree/main)



