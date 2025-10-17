import re

import chex
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

from . import rssm

f32 = jnp.float32
i32 = jnp.int32
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)
sample = lambda xs: jax.tree.map(lambda x: x.sample(nj.seed()), xs)
prefix = lambda xs, p: {f'{p}/{k}': v for k, v in xs.items()}
concat = lambda xs, a: jax.tree.map(lambda *x: jnp.concatenate(x, a), *xs)
isimage = lambda s: s.dtype == np.uint8 and len(s.shape) == 3


class BisimAgent(embodied.jax.Agent):
  """
  Bisimulation DreamerV3 Agent.
  
  """

  banner = [
      r"---  ___                           __   ______ ---",
      r"--- |   \ _ _ ___ __ _ _ __  ___ _ \ \ / /__ / ---",
      r"--- | |) | '_/ -_) _` | '  \/ -_) '/\ V / |_ \ ---",
      r"--- |___/|_| \___\__,_|_|_|_\___|_|  \_/ |___/ ---",
      r"---------------------bisim------------------------",
  ]

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    self.act_space = act_space
    self.config = config

    exclude = ('is_first', 'is_last', 'is_terminal', 'reward')
    enc_space = {k: v for k, v in obs_space.items() if k not in exclude}
    #dec_space = {k: v for k, v in obs_space.items() if k not in exclude}
    self.enc = {
        'simple': rssm.Encoder,
    }[config.enc.typ](enc_space, **config.enc[config.enc.typ], name='enc')
    self.dyn = {
        'rssm': rssm.RSSM,
    }[config.dyn.typ](act_space, **config.dyn[config.dyn.typ], name='dyn')
    # self.dec = {
    #     'simple': rssm.Decoder,
    # }[config.dec.typ](dec_space, **config.dec[config.dec.typ], name='dec')
    self.dec = None

    self.feat2tensor = lambda x: jnp.concatenate([
        nn.cast(x['deter']),
        nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1)))], -1)

    scalar = elements.Space(np.float32, ())
    binary = elements.Space(bool, (), 0, 2)
    self.rew = embodied.jax.MLPHead(scalar, **config.rewhead, name='rew')
    self.con = embodied.jax.MLPHead(binary, **config.conhead, name='con')

    d1, d2 = config.policy_dist_disc, config.policy_dist_cont
    outs = {k: d1 if v.discrete else d2 for k, v in act_space.items()}
    self.pol = embodied.jax.MLPHead(
        act_space, outs, **config.policy, name='pol')

    self.val = embodied.jax.MLPHead(scalar, **config.value, name='val')
    self.slowval = embodied.jax.SlowModel(
        embodied.jax.MLPHead(scalar, **config.value, name='slowval'),
        source=self.val, **config.slowvalue)

    self.retnorm = embodied.jax.Normalize(**config.retnorm, name='retnorm')
    self.valnorm = embodied.jax.Normalize(**config.valnorm, name='valnorm')
    self.advnorm = embodied.jax.Normalize(**config.advnorm, name='advnorm')

    self.modules = [
        self.dyn, self.rew, self.con, self.pol, self.val,self.enc]
    self.opt = embodied.jax.Optimizer(
        self.modules, self._make_opt(**config.opt), summary_depth=1,
        name='opt')
    #optimizer enc (if we separete it)
    # self.enc_opt = embodied.jax.Optimizer(
    #       [self.enc], self._make_opt(**config.opt), summary_depth=1,
    #      name='enc_opt')

    scales = self.config.loss_scales.copy()
    rec = scales.pop('rec')
    # scales.update({k: rec for k in dec_space})
    
    # Ensure bisim loss is in scales
    if 'bisim' not in scales:
      scales['bisim'] = getattr(config, 'bisim_scale', 1.0)
    
    
    self.scales = scales

  @property
  def policy_keys(self):
    return '^(enc|dyn|pol)/'

  @property
  def ext_space(self):
    spaces = {}
    spaces['consec'] = elements.Space(np.int32)
    spaces['stepid'] = elements.Space(np.uint8, 20)
    if self.config.replay_context:
      spaces.update(elements.tree.flatdict(dict(
          enc=self.enc.entry_space,
          dyn=self.dyn.entry_space)))
      # Note: dec removed in bisim agent since we don't use decoder
    return spaces

  def init_policy(self, batch_size):
    zeros = lambda x: jnp.zeros((batch_size, *x.shape), x.dtype)
    return (
        self.enc.initial(batch_size),
        self.dyn.initial(batch_size),
        jax.tree.map(zeros, self.act_space))

  def init_train(self, batch_size):
    return self.init_policy(batch_size)

  def init_report(self, batch_size):
    return self.init_policy(batch_size)

  def policy(self, carry, obs, mode='train'):
    (enc_carry, dyn_carry, prevact) = carry
    kw = dict(training=False, single=True)
    reset = obs['is_first']
    enc_carry, enc_entry, tokens = self.enc(enc_carry, obs, reset, **kw)
    dyn_carry, dyn_entry, feat = self.dyn.observe(
        dyn_carry, tokens, prevact, reset, **kw)
    # Note: No decoder in bisim agent - we focus on representation learning
    policy = self.pol(self.feat2tensor(feat), bdims=1)
    act = sample(policy)
    out = {}
    out['finite'] = elements.tree.flatdict(jax.tree.map(
        lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
        dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act)))
    carry = (enc_carry, dyn_carry, act)
    if self.config.replay_context:
      out.update(elements.tree.flatdict(dict(
          enc=enc_entry, dyn=dyn_entry)))
    return carry, act, out

  def train(self, carry, data):
    carry, obs, prevact, stepid = self._apply_replay_context(carry, data)
    #metrics_enc, aux = self.enc_opt(self.bisim_loss, carry, obs, prevact, training=True, has_aux=True)
    # metrics_enc contains 'enc_opt/...' prefixed metrics from Optimizer
    # aux contains (carry, metrics) from bisim_loss_only; extract the metrics if you need them
    #_, bisim_metrics = aux

    metrics, (carry, entries, outs, mets) = self.opt(
        self.loss, carry, obs, prevact, training=True, has_aux=True)
    metrics.update(mets)
  
    # log or merge metrics
    # if 'bisim_loss' in bisim_metrics:
    #    metrics['loss/bisim'] = bisim_metrics['bisim_loss'].mean()
    self.slowval.update()
    outs = {}
    if self.config.replay_context:
      updates = elements.tree.flatdict(dict(
          stepid=stepid, enc=entries[0], dyn=entries[1]))
      B, T = obs['is_first'].shape
      assert all(x.shape[:2] == (B, T) for x in updates.values()), (
          (B, T), {k: v.shape for k, v in updates.items()})
      outs['replay'] = updates
    # if self.config.replay.fracs.priority > 0:
    #   outs['replay']['priority'] = losses['model']
    carry = (*carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, outs, metrics


  def bisim_loss(self, carry, obs, prevact, training=True):
    """
    Compute only the bisimulation loss and return (loss, aux).
    aux can be whatever you want returned (e.g., metrics).
    This function should mirror how repfeat is produced in full loss,
    but return only the bisim term (as a scalar).
    """
    enc_carry, dyn_carry = carry
    reset = obs['is_first']
    B, T = reset.shape

    # Produce encoder and dynamics outputs similarly to full loss:
    enc_carry, enc_entries, tokens = self.enc(enc_carry, obs, reset, training)
    # dyn.observe yields repfeat used for bisim
    dyn_carry, dyn_entries, repfeat = self.dyn.observe(dyn_carry, sg(tokens), prevact, reset, training)
    
    
    # Convert repfeat to tensor (same helper used elsewhere)
    h = sg(self.feat2tensor(repfeat)) # shape [B, T, D]

    # Build permuted targets and detach them so targets are fixed
    perm = jax.random.permutation(nj.seed(), B)
    h2 = sg(h[perm])                 # detached permuted representation


    # reward targets and detach if needed
    rew = obs['reward']
    rew2 = rew[perm]
    
    z_dist = jnp.abs(h- h2)   # per-dimension distance
    r_dist = jnp.abs(rew-rew2)   # reward distance (detached)
  

    # Combine into bisim target and detach target
    discount = getattr(self.config, 'bisim_discount', 0.99)
    # If transition_dist exists, include it; else only reward
    # bisimilarity = r_dist + discount * jnp.mean(transition_dist, axis=-1)
    #I"M SURE WHICH IMPLEMENTATION OF USEFUL
    bisimilarity = r_dist + discount * jnp.mean(z_dist, axis=-1)
    bisimilarity = sg(bisimilarity)
    # Compare representation distance (reduce along feature dim, mean over time/batch):
    enc_dist = jnp.mean(jnp.abs(tokens - tokens[perm]), axis=-1)  # [B, T]
    loss = jnp.mean((bisimilarity -  enc_dist ) ** 2)  # scalar

    metrics = {'bisim_loss': loss, 'bisim_rep_rms': jnp.mean(z_dist)}

    # aux should match Optimizer expected aux shape if using has_aux=True
    return loss , (carry, metrics)

  # def   update_encoder_bisim(self, repfeat, act, rew, training=True):
  #   """
  #   Encoder bisimulation loss adapted to match deep_bisim4control's update_encoder:
  #   - Uses the learned dynamics self.dyn to produce one-step predicted next features
  #     deterministically (via the RSSM core + prior logits -> expected stoch).
  #   - Flattens (batch, time) into a single axis, forms a random permutation over
  #     that flattened axis, and computes smooth-L1 distances per-dimension.
  #   - Builds bisimulation target r_dist + gamma * transition_dist and trains the
  #     encoder so that representation distances match that target.
  #   - Returns a per-timestep loss array shaped (B, T) with a zero column appended
  #     for the final timestep (to fit the rest of the loss aggregation which
  #     expects (B, T) shapes).
  #   Notes:
  #     - Gradients are allowed to flow into self.dyn and self.rew from this loss
  #       (so the bisim metric can be used to update models), as requested.
  #     - repfeat is expected to be the "current" features (typically passed as
  #       repfeat[:, :-1] by the caller). If the final timestep was kept, the
  #       function will still pad an extra zero column.
  #   """
  #   # Smooth L1 (Huber-like) matching PyTorch F.smooth_l1_loss default beta=1.0
  #   def smooth_l1(a, b, beta=1.0):
  #     d = a - b
  #     ad = jnp.abs(d)
  #     return jnp.where(ad < beta, 0.5 * d * d / beta, ad - 0.5 * beta)

  #   # repfeat is a pytree with 'deter' [B, T, deter] and 'stoch' [B, T, stoch, classes]
  #   # act is a dict of actions with leading dims [B, T, ...]
  #   h = self.feat2tensor(repfeat)  # [B, T, D]
  #   B, T, D = h.shape

  #   # Nothing to do if no time dimension
  #   if B == 0 or T == 0:
  #     return jnp.zeros((B, T), dtype=jnp.float32)

  #   # Flatten across batch and time to create N = B * T examples
  #   flat_h = h.reshape((-1, D))               # [N, D]
  #   flat_rew = rew.reshape((-1,))             # [N]

  #   # Flatten actions for feeding into RSSM core (actions are pytrees)
  #   flat_act = jax.tree_map(lambda x: x.reshape((-1, *x.shape[2:])), act)

  #   # Build RSSM start carries from repfeat by flattening along batch/time
  #   starts = jax.tree_map(lambda x: x.reshape((-1, *x.shape[2:])), repfeat)

  #   # Build an action embedding as RSSM._core expects action already embedded by DictConcat in observe
  #   act_emb = nn.DictConcat(self.act_space, 1)(flat_act)

  #   # Use deterministic one-step prediction via RSSM core + prior logits -> expected stoch
  #   # deter_next = _core(deter, stoch, action_emb); prior_logits = _prior(deter_next)
  #   deter_flat = starts['deter']  # [N, deter]
  #   stoch_flat = starts['stoch']  # [N, stoch, classes] or similar

  #   # _core expects stoch shaped [..., stoch, classes]; it reshapes internally.
  #   deter_next = self.dyn._core(deter_flat, stoch_flat, act_emb)  # [N, deter]
  #   prior_logits = self.dyn._prior(deter_next)  # [N, stoch, classes]

  #   # Convert prior_logits to expected stoch (probabilities) deterministically via softmax
  #   expected_stoch = jax.nn.softmax(prior_logits, axis=-1)  # [N, stoch, classes]

  #   # Construct predicted feature pytree and convert to tensor
  #   pred_feat = dict(deter=deter_next, stoch=expected_stoch, logit=prior_logits)
  #   pred_h = self.feat2tensor(pred_feat).reshape((-1, D))  # [N, D]

  #   #normalizatiom
  #   h_mean = jnp.mean(flat_h, axis=0, keepdims=True)
  #   h_std = jnp.std(flat_h, axis=0, keepdims=True) + 1e-6
  #   print(flat_rew)
  #   flat_h= (flat_h - h_mean) / h_std

  #   pred_mean = jnp.mean(pred_h , axis=0, keepdims=True)
  #   pred_std = jnp.std(pred_h , axis=0, keepdims=True) + 1e-6
  #   pred_h = (pred_h  - pred_mean) / pred_std

  #   rew_mean = jnp.mean(flat_rew , axis=0, keepdims=True)
  #   rew_std = jnp.std(flat_rew, axis=0, keepdims=True) + 1e-6
  #   flat_rew = (flat_rew - rew_mean) / rew_std

  #   # Random permutation across flattened examples to form pairs
  #   N = flat_h.shape[0]
  #   perm = jax.random.permutation(nj.seed(), N)
  #   flat_h2 = flat_h[perm]
  #   pred_h2 = pred_h[perm]
  #   flat_rew2 = flat_rew[perm]

  #   # Elementwise (per-dim) smooth_l1 distances
  #   z_dist = smooth_l1(flat_h, flat_h2)                 # [N, D]
  #   transition_dist = smooth_l1(pred_h, pred_h2)        # [N, D]
  #   r_dist = smooth_l1(flat_rew[:, None], flat_rew2[:, None])  # [N, 1]

  #   discount = getattr(self.config, 'bisim_discount', 0.99)
  #   # Broadcast reward across representation dims and build bisim target
  #   bisim_target = r_dist + discount * transition_dist  # [N, D]

  #   # Per-dim squared error and mean across dims -> scalar per flattened sample
  #   per_dim_err = (z_dist - bisim_target) ** 2          # [N, D]
  #   per_sample_err = jnp.mean(per_dim_err, axis=-1)     # [N]

  #   # Reshape back to (B, T) and pad a zero final column if needed by the caller
  #   # Here repfeat was typically repfeat_now (i.e. original T was sequence length - 1),
  #   # but to be safe we pad a zero column so returned shape always matches original (B, T)
  #   # Determine B_orig and T_orig from rew shape passed by caller:
  #   B_orig = rew.shape[0]
  #   T_orig = rew.shape[1]
  #   # per_sample_err has N = B_orig * T_orig
  #   loss_per_timestep = per_sample_err.reshape((B_orig, T_orig))
  #   # If the caller passed truncated sequence (e.g., repfeat_now with T-1), we still return (B, T)
  #   # so no further padding here; the caller slices appropriately before passing.
  #   return loss_per_timestep


  # def update_transition_reward_model_loss(self, repfeat, act, next_repfeat, rew, training=True):
  #   """
  #   Transition + reward model loss that uses:
  #     - deterministic RSSM one-step prediction (core + prior -> expected stoch)
  #     - MSE/NLL-style transition loss between predicted features and next posterior features
  #     - Reward prediction loss from predicted features using self.rew
  #   Returns per-timestep loss shaped (B, T) with last timestep typically zero (if repfeat excludes final).
  #   Gradients flow into self.dyn and self.rew so both can be trained using this loss.
  #   """
  #   # repfeat: pytree current features [B, T, ...]
  #   # next_repfeat: pytree next features [B, T, ...] (usually aligned so both lengths equal and next_repfeat corresponds to t+1)
  #   h = self.feat2tensor(repfeat)  # [B, T, D]
  #   h_next = self.feat2tensor(next_repfeat)  # [B, T, D]
  #   B, T, D = h.shape

  #   if B == 0 or T == 0:
  #     return jnp.zeros((B, T), dtype=jnp.float32)

  #   # Flatten across batch/time
  #   flat_h = h.reshape((-1, D))         # [N, D]
  #   flat_h_next = h_next.reshape((-1, D))  # [N, D]
  #   flat_rew = rew.reshape((-1,))       # [N]

  #   # Flatten actions
  #   flat_act = jax.tree_map(lambda x: x.reshape((-1, *x.shape[2:])), act)

  #   # Build action embedding as RSSM.observe would do
  #   act_emb = nn.DictConcat(self.act_space, 1)(flat_act)

  #   # Build starts from repfeat flattened
  #   starts = jax.tree_map(lambda x: x.reshape((-1, *x.shape[2:])), repfeat)
  #   deter_flat = starts['deter']
  #   stoch_flat = starts['stoch']

  #   # Predict next deter via RSSM core and prior logits => expected stoch
  #   deter_next = self.dyn._core(deter_flat, stoch_flat, act_emb)  # [N, deter]
  #   prior_logits = self.dyn._prior(deter_next)                    # [N, stoch, classes]
  #   expected_stoch = jax.nn.softmax(prior_logits, axis=-1)        # [N, stoch, classes]

  #   pred_feat = dict(deter=deter_next, stoch=expected_stoch, logit=prior_logits)
  #   pred_h = self.feat2tensor(pred_feat).reshape((-1, D))         # [N, D]

  #   # Transition loss: use MSE between predicted embedding and next posterior embedding
  #   trans_per_dim = (pred_h - flat_h_next) ** 2                  # [N, D]
  #   trans_loss_per_sample = jnp.mean(trans_per_dim, axis=-1)     # [N]

  #   # Reward loss: predict reward from predicted embedding (same API as used elsewhere)
  #   #print(self.feat2tensor(pred_feat).shape, self.feat2tensor(repfeat).shape,)
  #   pred_rew = self.rew(self.feat2tensor(pred_feat).reshape((-1, T, D)), 2).pred().reshape((-1,))         # [N]
  #   reward_loss_per_sample = (pred_rew - flat_rew) ** 2          # [N]

  #   # Combine per-sample losses
  #   per_sample_loss = trans_loss_per_sample + reward_loss_per_sample  # [N]

  #   # Reshape to (B, T)
  #   loss_per_timestep = per_sample_loss.reshape((B, T))

  #   return loss_per_timestep
 
  def loss(self, carry, obs, prevact, training=True):
    enc_carry, dyn_carry = carry
    reset = obs['is_first']
    B, T = reset.shape
    losses = {}
    metrics = {}

    # World model
    enc_carry, enc_entries, tokens = self.enc(
        enc_carry, obs, reset, training)
    dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
        dyn_carry, tokens, prevact, reset, training)
    losses.update(los)
    metrics.update(mets)
    #dec_carry, dec_entries, recons = self.dec(
    #    dec_carry, repfeat, reset, training)
    inp = sg(self.feat2tensor(repfeat), skip=self.config.reward_grad)
    losses['rew'] = self.rew(inp, 2).loss(obs['reward'])
    con = f32(~obs['is_terminal'])
    if self.config.contdisc:
      con *= 1 - 1 / self.config.horizon
    losses['con'] = self.con(self.feat2tensor(repfeat), 2).loss(con)
    #for key, recon in recons.items():
    #  space, value = self.obs_space[key], obs[key]
    #  assert value.dtype == space.dtype, (key, space, value.dtype)
    #  target = f32(value) / 255 if isimage(space) else value
    #  losses[key] = recon.loss(sg(target))

    B, T = reset.shape
    shapes = {k: v.shape for k, v in losses.items()}
    assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)

    # Imagination
    K = min(self.config.imag_last or T, T)
    H = self.config.imag_length
    starts = self.dyn.starts(dyn_entries, dyn_carry, K)
    policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1))
    _, imgfeat, imgprevact = self.dyn.imagine(starts, policyfn, H, training)
    first = jax.tree.map(
        lambda x: x[:, -K:].reshape((B * K, 1, *x.shape[2:])), repfeat)
    imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(imgfeat)], 1)
    lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
    lastact = jax.tree.map(lambda x: x[:, None], lastact)
    imgact = concat([imgprevact, lastact], 1)
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgfeat))
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgact))
    inp = self.feat2tensor(imgfeat)
    los, imgloss_out, mets = imag_loss(
        imgact,
        self.rew(inp, 2).pred(),
        self.con(inp, 2).prob(1),
        self.pol(inp, 2),
        self.val(inp, 2),
        self.slowval(inp, 2),
        self.retnorm, self.valnorm, self.advnorm,
        update=training,
        contdisc=self.config.contdisc,
        horizon=self.config.horizon,
        **self.config.imag_loss)
    losses.update({k: v.mean(1).reshape((B, K)) for k, v in los.items()})
    metrics.update(mets)

    # Enhanced bisim loss with encoder and transition model updates 
    #bisim_coef = getattr(self.config, 'bisim_coef', 0.5)
    #transition_coef = getattr(self.config, 'transition_coef', 1.0)
    
    # repfeat_now = jax.tree.map(lambda x: x[:, :-1], repfeat)
    # repfeat_next = jax.tcree.map(lambda x: x[:, 1:], repfeat)
    # act_now = {k: v[:, :-1] for k, v in prevact.items()}
    # rew_now = obs['reward'][:, :-1]
    
    #bisim
    losses['bisim'], _= self.bisim_loss(carry, obs, prevact)
    
    # Encoder bisim loss (returns shape (B, T-1) or (B, T) depending on input); we expect (B, T-1)
    # encoder_bisim_loss = self.update_encoder_bisim(repfeat_now, act_now, rew_now, training)
    
    # # Transition and reward model loss 
    # transition_reward_loss = self.update_transition_reward_model_loss(
    #     repfeat_now, act_now, repfeat_next, rew_now, training)
    
    # # Combine bisimilarity-related losses 
    # # both are (B, T) or (B, T-1) shaped consistent with loss aggregation; ensure they are (B, T)
    # # If they are (B, T-1), pad to (B, T)
    # def pad_to_T(x):
    #   if x.shape[1] == T:
    #     return x
    #   else:
    #     pad = jnp.zeros((x.shape[0], 1), dtype=x.dtype)
    #     return jnp.concatenate([x, pad], axis=1)

    # encoder_bisim_loss = pad_to_T(encoder_bisim_loss)
    # transition_reward_loss = pad_to_T(transition_reward_loss)

    # total_bisim_loss = bisim_coef * encoder_bisim_loss + transition_coef * transition_reward_loss
    # losses['bisim'] = total_bisim_loss
    

    # Replay
    if self.config.repval_loss:
      feat = sg(repfeat, skip=self.config.repval_grad)
      last, term, rew = [obs[k] for k in ('is_last', 'is_terminal', 'reward')]
      boot = imgloss_out['ret'][:, 0].reshape(B, K)
      feat, last, term, rew, boot = jax.tree.map(
          lambda x: x[:, -K:], (feat, last, term, rew, boot))
      inp = self.feat2tensor(feat)
      los, reploss_out, mets = repl_loss(
          last, term, rew, boot,
          self.val(inp, 2),
          self.slowval(inp, 2),
          self.valnorm,
          update=training,
          horizon=self.config.horizon,
          **self.config.repl_loss)
      losses.update(los)
      metrics.update(prefix(mets, 'reploss'))

    assert set(losses.keys()) == set(self.scales.keys()), (
        sorted(losses.keys()), sorted(self.scales.keys()))
    metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
    loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])

    carry = (enc_carry, dyn_carry)
    entries = (enc_entries, dyn_entries)
    outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}
    return loss, (carry, entries, outs, metrics)

  def report(self, carry, data):
    if not self.config.report:
      return carry, {}

    carry, obs, prevact, _ = self._apply_replay_context(carry, data)
    (enc_carry, dyn_carry) = carry
    B, T = obs['is_first'].shape
    RB = min(6, B)
    metrics = {}

    # Train metrics
    _, (new_carry, entries, outs, mets) = self.loss(
        carry, obs, prevact, training=False)
    mets.update(mets)

    # Grad norms
    if self.config.report_gradnorms:
      for key in self.scales:
        try:
          lossfn = lambda data, carry: self.loss(
              carry, obs, prevact, training=False)[1][2]['losses'][key].mean()
          grad = nj.grad(lossfn, self.modules)(data, carry)[-1]
          metrics[f'gradnorm/{key}'] = optax.global_norm(grad)
        except KeyError:
          print(f'Skipping gradnorm summary for missing loss: {key}')

    # Open loop
    firsthalf = lambda xs: jax.tree.map(lambda x: x[:RB, :T // 2], xs)
    secondhalf = lambda xs: jax.tree.map(lambda x: x[:RB, T // 2:], xs)
    dyn_carry = jax.tree.map(lambda x: x[:RB], dyn_carry)
    dyn_carry, _, obsfeat = self.dyn.observe(
        dyn_carry, firsthalf(outs['tokens']), firsthalf(prevact),
        firsthalf(obs['is_first']), training=False)
    _, imgfeat, _ = self.dyn.imagine(
        dyn_carry, secondhalf(prevact), length=T - T // 2, training=False)
    
    # Video preds

    # obs_tensor = self.feat2tensor(obsfeat)  # [RB, T//2, feat_dim]
    # img_tensor = self.feat2tensor(imgfeat)  # [RB, T//2, feat_dim]
    # feat_tensor = jnp.concatenate([obs_tensor, img_tensor], axis=1)  # [RB, T, feat_dim]
    # feat_dim = feat_tensor.shape[-1]
    # height = int(feat_dim**0.5)#jnp.sqrt(feat_dim).astype(int)
    # width = feat_dim // height
    
    # # Truncate to make perfect rectangle
    # used_dims = height * width
    # features = feat_tensor.shape[-1]  # static or ShapedArray: used to build mask length
    # # build a boolean mask shape (1,1,features) where first used_dims are True
    # mask = (jnp.arange(features)[None, None, :] < used_dims).astype(feat_tensor.dtype)
    # # zero out the trailing dims and then reshape
    # feat_masked = feat_tensor * mask
    # feat_reshaped = feat_masked.reshape(RB, T, height, width, 1)
    
    # # Normalize to [0, 255] for visualization
    # feat_min = feat_reshaped.min(axis=(2, 3), keepdims=True)
    # feat_max = feat_reshaped.max(axis=(2, 3), keepdims=True)
    # feat_norm = (feat_reshaped - feat_min) / (feat_max - feat_min + 1e-8)
    # feat_img = (feat_norm * 255).astype(jnp.uint8)
    
    # # Convert to RGB
    # feat_img = jnp.repeat(feat_img, 3, axis=-1)
    
    # # Add borders (green for obs, red for imagination)
    # feat_img = jnp.pad(feat_img, [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]])
    # mask = jnp.zeros(feat_img.shape, bool).at[:, :, 2:-2, 2:-2, :].set(True)
    # border = jnp.full((T, 3), jnp.array([0, 255, 0]), jnp.uint8)
    # border = border.at[T // 2:].set(jnp.array([255, 0, 0], jnp.uint8))
    # feat_img = jnp.where(mask, feat_img, border[None, :, None, None, :])
    
    # # Add spacing and create grid
    # feat_img = jnp.concatenate([feat_img, 0 * feat_img[:, :10]], 1)
    # B, T, H, W, C = feat_img.shape
    # grid = feat_img.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
    # metrics[f'openloop/dyn_features'] = grid
    carry = (*new_carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, metrics

  def _apply_replay_context(self, carry, data):
    (enc_carry, dyn_carry, prevact) = carry
    carry = (enc_carry, dyn_carry)
    stepid = data['stepid']
    obs = {k: data[k] for k in self.obs_space}
    prepend = lambda x, y: jnp.concatenate([x[:, None], y[:, :-1]], 1)
    prevact = {k: prepend(prevact[k], data[k]) for k in self.act_space}
    if not self.config.replay_context:
      return carry, obs, prevact, stepid

    K = self.config.replay_context
    nested = elements.tree.nestdict(data)
    entries = [nested.get(k, {}) for k in ('enc', 'dyn')]
    lhs = lambda xs: jax.tree.map(lambda x: x[:, :K], xs)
    rhs = lambda xs: jax.tree.map(lambda x: x[:, K:], xs)
    rep_carry = (
        self.enc.truncate(lhs(entries[0]), enc_carry),
        self.dyn.truncate(lhs(entries[1]), dyn_carry),
        #self.dec.truncate(lhs(entries[2]), dec_carry)
        )
    rep_obs = {k: rhs(data[k]) for k in self.obs_space}
    rep_prevact = {k: data[k][:, K - 1: -1] for k in self.act_space}
    rep_stepid = rhs(stepid)

    first_chunk = (data['consec'][:, 0] == 0)
    carry, obs, prevact, stepid = jax.tree.map(
        lambda normal, replay: nn.where(first_chunk, replay, normal),
        (carry, rhs(obs), rhs(prevact), rhs(stepid)),
        (rep_carry, rep_obs, rep_prevact, rep_stepid))
    return carry, obs, prevact, stepid

  def _make_opt(
      self,
      lr: float = 4e-5,
      agc: float = 0.3,
      eps: float = 1e-20,
      beta1: float = 0.9,
      beta2: float = 0.999,
      momentum: bool = True,
      nesterov: bool = False,
      wd: float = 0.0,
      wdregex: str = r'/kernel$',
      schedule: str = 'const',
      warmup: int = 1000,
      anneal: int = 0,
  ):
    chain = []
    chain.append(embodied.jax.opt.clip_by_agc(agc))
    chain.append(embodied.jax.opt.scale_by_rms(beta2, eps))
    chain.append(embodied.jax.opt.scale_by_momentum(beta1, nesterov))
    if wd:
      assert not wdregex[0].isnumeric(), wdregex
      pattern = re.compile(wdregex)
      wdmask = lambda params: {k: bool(pattern.search(k)) for k in params}
      chain.append(optax.add_decayed_weights(wd, wdmask))
    assert anneal > 0 or schedule == 'const'
    if schedule == 'const':
      sched = optax.constant_schedule(lr)
    elif schedule == 'linear':
      sched = optax.linear_schedule(lr, 0.1 * lr, anneal - warmup)
    elif schedule == 'cosine':
      sched = optax.cosine_decay_schedule(lr, anneal - warmup, 0.1 * lr)
    else:
      raise NotImplementedError(schedule)
    if warmup:
      ramp = optax.linear_schedule(0.0, lr, warmup)
      sched = optax.join_schedules([ramp, sched], [warmup])
    chain.append(optax.scale_by_learning_rate(sched))
    return optax.chain(*chain)


def imag_loss(
    act, rew, con,
    policy, value, slowvalue,
    retnorm, valnorm, advnorm,
    update,
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    slowreg=1.0,
):
  losses = {}
  metrics = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 if contdisc else 1 - 1 / horizon
  weight = jnp.cumprod(disc * con, 1) / disc
  last = jnp.zeros_like(con)
  term = 1 - con
  ret = lambda_return(last, term, rew, tarval, tarval, disc, lam)

  roffset, rscale = retnorm(ret, update)
  adv = (ret - tarval[:, :-1]) / rscale
  aoffset, ascale = advnorm(adv, update)
  adv_normed = (adv - aoffset) / ascale
  logpi = sum([v.logp(sg(act[k]))[:, :-1] for k, v in policy.items()])
  ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}
  policy_loss = sg(weight[:, :-1]) * -(
      logpi * sg(adv_normed) + actent * sum(ents.values()))
  losses['policy'] = policy_loss

  voffset, vscale = valnorm(ret, update)
  tar_normed = (ret - voffset) / vscale
  tar_padded = jnp.concatenate([tar_normed, 0 * tar_normed[:, -1:]], 1)
  losses['value'] = sg(weight[:, :-1]) * (
      value.loss(sg(tar_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  ret_normed = (ret - roffset) / rscale
  metrics['adv'] = adv.mean()
  metrics['adv_std'] = adv.std()
  metrics['adv_mag'] = jnp.abs(adv).mean()
  metrics['rew'] = rew.mean()
  metrics['con'] = con.mean()
  metrics['ret'] = ret_normed.mean()
  metrics['val'] = val.mean()
  metrics['tar'] = tar_normed.mean()
  metrics['weight'] = weight.mean()
  metrics['slowval'] = slowval.mean()
  metrics['ret_min'] = ret_normed.min()
  metrics['ret_max'] = ret_normed.max()
  metrics['ret_rate'] = (jnp.abs(ret_normed) >= 1.0).mean()
  for k in act:
    metrics[f'ent/{k}'] = ents[k].mean()
    if hasattr(policy[k], 'minent'):
      lo, hi = policy[k].minent, policy[k].maxent
      metrics[f'rand/{k}'] = (ents[k].mean() - lo) / (hi - lo)

  outs = {}
  outs['ret'] = ret
  return losses, outs, metrics


def repl_loss(
    last, term, rew, boot,
    value, slowvalue, valnorm,
    update=True,
    slowreg=1.0,
    slowtar=True,
    horizon=333,
    lam=0.95,
):
  losses = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 - 1 / horizon
  weight = f32(~last)
  ret = lambda_return(last, term, rew, tarval, boot, disc, lam)

  voffset, vscale = valnorm(ret, update)
  ret_normed = (ret - voffset) / vscale
  ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[:, -1:]], 1)
  losses['repval'] = weight[:, :-1] * (
      value.loss(sg(ret_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  outs = {}
  outs['ret'] = ret
  metrics = {}

  return losses, outs, metrics


def lambda_return(last, term, rew, val, boot, disc, lam):
  chex.assert_equal_shape((last, term, rew, val, boot))
  rets = [boot[:, -1]]
  live = (1 - f32(term))[:, 1:] * disc
  cont = (1 - f32(last))[:, 1:] * lam
  interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
  for t in reversed(range(live.shape[1])):
    rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
  return jnp.stack(list(reversed(rets))[:-1], 1)
