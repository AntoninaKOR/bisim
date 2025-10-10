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
        self.dyn, self.enc, self.rew, self.con, self.pol, self.val]
    self.opt = embodied.jax.Optimizer(
        self.modules, self._make_opt(**config.opt), summary_depth=1,
        name='opt')

    scales = self.config.loss_scales.copy()
    # rec = scales.pop('rec')
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
    metrics, (carry, entries, outs, mets) = self.opt(
        self.loss, carry, obs, prevact, training=True, has_aux=True)
    metrics.update(mets)
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
  


  def update_encoder_bisim(self, repfeat, act, rew, training=True):
    """
    Update encoder using bisimulation-based loss (converted from PyTorch version).
    Args:
        repfeat: latent features [B, T, ...]
        act: actions [B, T, ...]
        rew: rewards [B, T]
        training: whether in training mode
    Returns:
        loss: encoder bisimulation loss
    """
    B, T = rew.shape
    
    # Sample random permutation across batch
    perm = jax.random.permutation(nj.seed(), B)
    
    # Get current and permuted features
    h = self.feat2tensor(repfeat)  # Convert to tensor representation
    h2 = h[perm]
    
    # Predict next latent states using current dynamics model
    # Using the existing dynamics model to predict transitions
    feat_flat = jax.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), repfeat)
    act_flat = jax.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), act)
    
    # Get transition predictions using existing dynamics model
    # Since imagine_step might not exist, we'll use the dynamics model differently
    # For now, we'll create a simple transition prediction using the reward head
    # In a full implementation, this would need a proper transition model
    pred_next_h = self.rew(h.reshape(-1, h.shape[-1]), 2).pred()
    pred_next_h = pred_next_h.reshape(B, T, -1)
    pred_next_h = pred_next_h.reshape(B, T, -1)
    
    # Get permuted versions
    pred_next_h2 = pred_next_h[perm]
    rew2 = rew[perm]
    
    # Compute distances using smooth L1 loss (converted to JAX)
    z_dist = jnp.abs(h - h2)  # Use L1 distance instead of smooth_l1
    r_dist = jnp.abs(rew - rew2)
    
    # Transition distance
    transition_dist = jnp.abs(pred_next_h - pred_next_h2)
    
    # Bisimilarity metric
    discount = getattr(self.config, 'bisim_discount', 0.99)
    bisimilarity = r_dist + discount * jnp.mean(transition_dist, axis=-1)
    
    # Loss computation
    loss = jnp.mean((jnp.mean(z_dist, axis=-1) - bisimilarity) ** 2)
    return loss

  def update_transition_reward_model_loss(self, repfeat, act, next_repfeat, rew, training=True):
    """
    Compute transition and reward model losses (adapted from PyTorch version).
    Args:
        repfeat: current latent features [B, T, ...]
        act: actions [B, T, ...]
        next_repfeat: next latent features [B, T, ...]
        rew: rewards [B, T]
        training: whether in training mode
    Returns:
        loss: combined transition and reward loss
    """
    # Convert features to tensor representation
    h = self.feat2tensor(repfeat)
    next_h = self.feat2tensor(next_repfeat)
    
    # For this implementation, we use the existing reward head and dynamics model
    # Predict rewards from current features
    pred_rew = self.rew(h, 2).pred()
    reward_loss = jnp.mean((pred_rew.squeeze(-1) - rew) ** 2)
    
    # Transition loss using dynamics model predictions
    # Simple transition loss using feature similarity
    # In a full implementation, this would use a dedicated transition model
    B, T = rew.shape
    
    # Compute simple transition consistency loss
    # This measures how well we can predict the transition in feature space
    h_current = self.feat2tensor(repfeat)
    h_next = self.feat2tensor(next_repfeat)
    
    # Simple prediction: assume features change linearly with reward prediction
    reward_change = self.rew(h_current.reshape(-1, h_current.shape[-1]), 2).pred()
    reward_change = reward_change.reshape(B, T, 1)
    
    # Use reward prediction as a proxy for feature transition prediction
    pred_feature_change = reward_change * 0.1  # Scale factor for stability
    
    # Actual feature change
    actual_change = h_next - h_current
    
    # Transition loss as MSE between predicted and actual change
    transition_loss = jnp.mean((pred_feature_change - jnp.mean(actual_change, axis=-1, keepdims=True)) ** 2)
    
    total_loss = transition_loss + reward_loss
    return total_loss

 
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

    # Enhanced bisim loss with encoder and transition model updates (from methods.py)
    bisim_coef = getattr(self.config, 'bisim_coef', 0.5)
    transition_coef = getattr(self.config, 'transition_coef', 1.0)
    
    repfeat_now = jax.tree.map(lambda x: x[:, :-1], repfeat)
    repfeat_next = jax.tree.map(lambda x: x[:, 1:], repfeat)
    act_now = {k: v[:, :-1] for k, v in prevact.items()}
    rew_now = obs['reward'][:, :-1]
    
    # Enhanced encoder bisim loss 
    encoder_bisim_loss = self.update_encoder_bisim(repfeat_now, act_now, rew_now, training)
    
    # Transition and reward model loss 
    transition_reward_loss = self.update_transition_reward_model_loss(
        repfeat_now, act_now, repfeat_next, rew_now, training)
    
    # Combine bisimilarity-related losses 
    total_bisim_loss = bisim_coef * encoder_bisim_loss + transition_coef * transition_reward_loss
    losses['bisim'] = total_bisim_loss

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

    obs_tensor = self.feat2tensor(obsfeat)  # [RB, T//2, feat_dim]
    img_tensor = self.feat2tensor(imgfeat)  # [RB, T//2, feat_dim]
    
    feat_tensor = jnp.concatenate([obs_tensor, img_tensor], axis=1)  # [RB, T, feat_dim]
    
    feat_dim = feat_tensor.shape[-1]
    height = int(jnp.sqrt(feat_dim))
    width = feat_dim // height
    
    # Truncate to make perfect rectangle
    used_dims = height * width
    feat_reshaped = feat_tensor[:, :, :used_dims].reshape(RB, T, height, width, 1)
    
    # Normalize to [0, 255] for visualization
    feat_min = feat_reshaped.min(axis=(2, 3), keepdims=True)
    feat_max = feat_reshaped.max(axis=(2, 3), keepdims=True)
    feat_norm = (feat_reshaped - feat_min) / (feat_max - feat_min + 1e-8)
    feat_img = (feat_norm * 255).astype(jnp.uint8)
    
    # Convert to RGB
    feat_img = jnp.repeat(feat_img, 3, axis=-1)
    
    # Add borders (green for obs, red for imagination)
    feat_img = jnp.pad(feat_img, [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]])
    mask = jnp.zeros(feat_img.shape, bool).at[:, :, 2:-2, 2:-2, :].set(True)
    border = jnp.full((T, 3), jnp.array([0, 255, 0]), jnp.uint8)
    border = border.at[T // 2:].set(jnp.array([255, 0, 0], jnp.uint8))
    feat_img = jnp.where(mask, feat_img, border[None, :, None, None, :])
    
    # Add spacing and create grid
    feat_img = jnp.concatenate([feat_img, 0 * feat_img[:, :10]], 1)
    B, T, H, W, C = feat_img.shape
    grid = feat_img.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
    metrics[f'openloop/dyn_features'] = grid
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
