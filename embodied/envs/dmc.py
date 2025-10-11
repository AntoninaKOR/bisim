import functools
import os

import elements
import embodied
import numpy as np
from dm_control import manipulation
from dm_control import suite
import dmc2gym
from . import from_dm
from dm_control import suite
from gym import core
from embodied.envs import distractors

class DMC(embodied.Env):

  DEFAULT_CAMERAS = dict(
      quadruped=2,
      rodent=4,
  )

  def __init__(
      self, env, repeat=1, size=(64, 64), proprio=True, image=True, img_source=None, camera=-1, total_frames=10000, bg_grayscale=True):
    if 'MUJOCO_GL' not in os.environ:
      os.environ['MUJOCO_GL'] = 'egl'
    if isinstance(env, str):
      domain, task = env.split('_', 1)
      if camera == -1:
        camera = self.DEFAULT_CAMERAS.get(domain, 0)
      if domain == 'cup':  # Only domain with multiple words.
        domain = 'ball_in_cup'
      if domain == 'manip':
        env = manipulation.load(task + '_vision')
      elif domain == 'rodent':
        # camera 0: topdown map
        # camera 2: shoulder
        # camera 4: topdown tracking
        # camera 5: eyes
        env = getattr(basic_rodent_2020, task)()
      else:
        env = suite.load(domain_name=domain, task_name=task)
    self._dmenv = env
    self._env = from_dm.FromDM(self._dmenv)
    self._env = embodied.wrappers.ActionRepeat(self._env, repeat)
    self._size = size
    self._proprio = proprio
    self._image = image
    self._camera = camera
    self._bg_source = None
    if img_source is not None:
      try:
        self._bg_source = distractors.make_bg_source(
            img_source, height=size[0], width=size[1],
            grayscale=bg_grayscale, total_frames=total_frames)
      except Exception as e:
        print("Warning: could not create bg_source:", e)
        self._bg_source = None

  @functools.cached_property
  def obs_space(self):
    basic = ('is_first', 'is_last', 'is_terminal', 'reward')
    spaces = self._env.obs_space.copy()
    if not self._proprio:
      spaces = {k: spaces[k] for k in basic}
    key = 'image' if self._image else 'log/image'
    spaces[key] = elements.Space(np.uint8, self._size + (3,))
    return spaces

  @functools.cached_property
  def act_space(self):
    return self._env.act_space

  def step(self, action):
    for key, space in self.act_space.items():
      if not space.discrete:
        assert np.isfinite(action[key]).all(), (key, action[key])
    obs = self._env.step(action)
    basic = ('is_first', 'is_last', 'is_terminal', 'reward')
    if not self._proprio:
      obs = {k: obs[k] for k in basic}
    key = 'image' if self._image else 'log/image'
    frame = self._dmenv.physics.render(*self._size, camera_id=self._camera)

    # Compose distractor background if available.
    if self._bg_source is not None:
      try:

        try:
          bg = self._bg_source.get_image(background=None)  # try to use signature with background
        except TypeError:
          bg = self._bg_source.get_image()
        # ensure shapes and types
        if bg.shape[:2] != frame.shape[:2]:
          from PIL import Image
          bg = np.array(Image.fromarray(bg).resize((frame.shape[1], frame.shape[0])).convert('RGB'), dtype=np.uint8)
        frame = frame.copy()
        # chroma-key mask heuristic (sky is often blue in dm_control scenes).
        mask = np.logical_and(frame[:, :, 2] > frame[:, :, 1], frame[:, :, 2] > frame[:, :, 0])
        frame[mask] = bg[mask]
      except Exception as e:
        print("Warning: bg compositing failed:", e)

    obs[key] = frame
    for key, space in self.obs_space.items():
      if np.issubdtype(space.dtype, np.floating):
        assert np.isfinite(obs[key]).all(), (key, obs[key])
    return obs
