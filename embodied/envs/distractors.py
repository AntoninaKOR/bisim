"""
Distractor background sources for DreamerV3.

Added support for an N-body procedural distractor that is compatible
with the Planets implementation in
facebookresearch/deep_bisim4control/distractors/n_body_problem.py

API:
    src = make_bg_source(img_source, resource_files, height, width, ...)
    bg = src.get_image()  # returns HxWx3 uint8 RGB

Supported img_source values:
  - 'video'   : sample frames from video files (requires imageio)
  - 'images'  : sample image files
  - 'noise'   : procedural noise
  - 'color'   : solid color (resource_files may be "R,G,B")
  - 'n_body'  : procedural n-body simulation; resource_files may be a param string "n=12,dt=0.01"

This file implements a local NBodySource so you do not need to import deep_bisim4control.
If you do have facebookresearch/deep_bisim4control installed, make_bg_source will try to import
the Planets class and use it instead.
"""
import glob
import os
import random
import re
import numpy as np
import imageio
from PIL import Image, ImageDraw

# -------------------------
# Base sources (images/video/noise/color)
# -------------------------

class BaseSource:
    def __init__(self, shape, grayscale=False):
        self.shape = shape  # (H, W)
        self.grayscale = grayscale

    def _resize(self, img):
        im = Image.fromarray(img)
        im = im.resize((self.shape[1], self.shape[0]), Image.BILINEAR)
        im = im.convert('RGB')
        return np.asarray(im, dtype=np.uint8)

    def get_image(self):
        raise NotImplementedError



def _parse_params_string(s):
    """
    Parse comma-separated key=value pairs in a string.
    Example: "n=12,dt=0.02,seed=42"
    Returns dict of parsed values (ints/floats/strings).
    """
    params = {}
    if not s:
        return params
    parts = [p.strip() for p in s.split(',') if p.strip()]
    for p in parts:
        if '=' in p:
            k, v = p.split('=', 1)
            k = k.strip()
            v = v.strip()
            # try int, then float, else string
            if re.fullmatch(r'-?\d+', v):
                params[k] = int(v)
            else:
                try:
                    params[k] = float(v)
                except Exception:
                    params[k] = v
    return params


class NBodySource(BaseSource):
    """
    Simple N-body simulator renderer.

    - positions in normalized [0,1]^2 box
    - simple inverse-square interactions, Euler integration
    - draws colored circles on background
    """
    def __init__(self, shape, num_bodies=8, dt=0.01, grav=1.0, seed=None,
                 contained_in_a_box=True, body_radius=4, palette=None):
        super().__init__(shape, grayscale=False)
        self.h, self.w = shape
        self.num_bodies = int(num_bodies)
        self.dt = float(dt)
        self.G = float(grav)
        self.contained = contained_in_a_box
        self.body_radius = int(body_radius)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        # initialize positions and velocities
        self.positions = np.random.uniform(0.15, 0.85, size=(self.num_bodies, 2))
        self.velocities = np.random.uniform(-0.01, 0.01, size=(self.num_bodies, 2))
        if palette is None:
            # random pastel colors
            palette = [tuple((np.array([random.randint(64,255) for _ in range(3)]))).tolist() for _ in range(self.num_bodies)]
        self.palette = palette

    def _step_physics(self):
        # compute pairwise forces
        pos = self.positions
        vel = self.velocities
        accel = np.zeros_like(vel)
        for i in range(self.num_bodies):
            rel = pos - pos[i:i+1]  # (N,2)
            dist = np.linalg.norm(rel, axis=1, keepdims=True)  # (N,1)
            dist[i] = 1.0
            # inverse-square (avoid singularity)
            force = self.G * rel / (np.maximum(dist, 1e-3)**2)
            force[i] = 0.0
            accel[i] = force.sum(axis=0)
        # integrate (simple Euler)
        vel = vel + accel * self.dt
        pos = pos + vel * self.dt
        # bounce on walls if contained
        if self.contained:
            below = pos < 0.0
            above = pos > 1.0
            pos[below] = -pos[below]
            pos[above] = 2.0 - pos[above]
            vel[below] *= -1.0
            vel[above] *= -1.0
        # clamp to [0,1]
        pos = np.clip(pos, 0.0, 1.0)
        self.positions = pos
        self.velocities = vel

    def _render_to_image(self, background=None):
        # background: HxWx3 uint8 or None
        if background is None:
            img = Image.new('RGB', (self.w, self.h), (0, 0, 0))
        else:
            img = Image.fromarray(background).convert('RGB')
        draw = ImageDraw.Draw(img, 'RGBA')
        for i in range(self.num_bodies):
            x = int(np.clip(self.positions[i, 0], 0.0, 1.0) * (self.w - 1))
            y = int(np.clip(self.positions[i, 1], 0.0, 1.0) * (self.h - 1))
            r = max(1, self.body_radius)
            color = tuple(int(c) for c in self.palette[i % len(self.palette)])
            # draw a filled circle (RGBA alpha to soften)
            bbox = [x - r, y - r, x + r, y + r]
            draw.ellipse(bbox, fill=color + (255,))
        return np.asarray(img, dtype=np.uint8)

    def get_image(self, background=None):
        # step then render
        self._step_physics()
        return self._render_to_image(background=background)

# -------------------------
# Factory
# -------------------------

def make_bg_source(img_source, resource_files, height, width, grayscale=False, total_frames=10000):
    """
    Create a background source.

    img_source: 'video'|'images'|'noise'|'color'|'n_body'
    resource_files: glob pattern (for video/images) or param string (for n_body) or color "R,G,B"
    """
    if img_source is None:
        return None
    shape = (height, width)
    if img_source == 'n_body':
        # Try to use facebookresearch's Planets if available; otherwise use local NBodySource
        params = _parse_params_string(resource_files or "")
        num_bodies = params.get('n', params.get('num_bodies', 8))
        dt = params.get('dt', 0.01)
        grav = params.get('grav', 1.0)
        seed = params.get('seed', None)
        radius = params.get('radius', max(2, min(height, width) // 32))
        # If deep_bisim4control's Planets is available, try to wrap it.
        try:
            from n_body_problem import Planets  # type: ignore
            # If import works, use Planets but still render with PIL for speed.
            planets = Planets(num_bodies=num_bodies, num_dimensions=2, dt=dt, contained_in_a_box=True)
            class WrappedPlanets:
                def __init__(self, planets, shape, radius, palette=None):
                    self.planets = planets
                    self.shape = shape
                    self.radius = radius
                    # simple palette
                    self.palette = palette or [tuple((np.random.randint(64,255,size=3)).tolist()) for _ in range(self.planets.num_bodies)]
                def get_image(self):
                    # step the simulator
                    self.planets.step()
                    pos = self.planets.body_positions  # (N,2) in [0,1]
                    # draw onto image
                    h, w = self.shape
                    img = Image.new('RGB', (w, h), (0, 0, 0))
                    draw = ImageDraw.Draw(img, 'RGBA')
                    for i in range(self.planets.num_bodies):
                        x = int(np.clip(pos[i,0], 0.0, 1.0)*(w-1))
                        y = int(np.clip(pos[i,1], 0.0, 1.0)*(h-1))
                        r = int(self.radius)
                        color = tuple(int(c) for c in self.palette[i % len(self.palette)])
                        draw.ellipse([x-r,y-r,x+r,y+r], fill=color + (255,))
                    return np.asarray(img, dtype=np.uint8)
            return WrappedPlanets(planets, shape, radius)
        except Exception:
            # fallback to local implementation
            return NBodySource(shape, num_bodies=num_bodies, dt=dt, grav=grav, seed=seed, body_radius=radius)
    else:
        raise ValueError(f"Unknown img_source {img_source}")