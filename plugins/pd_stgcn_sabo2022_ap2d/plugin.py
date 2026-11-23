"""Sabo et al. 2022, the 2D model for AlphaPose joints. Same study as pd_stgcn_sabo2022 (3D Kinect model); shows that a study's model can be swapped."""
from pathlib import Path

from shared.sabo2d import make

PLUGIN = make("alphapose", "AlphaPose", Path(__file__).parent, "ap2d")
