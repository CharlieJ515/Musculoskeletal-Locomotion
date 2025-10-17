from collections import Counter
from pathlib import Path
from dataclasses import dataclass, fields
from typing import Dict, Iterable, List, Optional, Tuple, ClassVar
from math import isnan, isfinite

import torch
import numpy as np
from numpy.typing import NDArray
import torch
import opensim


MUSCLE_ORDER = (
    "abd_r", "add_r", "hamstrings_r", "bifemsh_r", "glut_max_r",
    "iliopsoas_r", "rect_fem_r", "vasti_r", "gastroc_r", "soleus_r", "tib_ant_r",
    "abd_l", "add_l", "hamstrings_l", "bifemsh_l", "glut_max_l",
    "iliopsoas_l", "rect_fem_l", "vasti_l", "gastroc_l", "soleus_l", "tib_ant_l",
)


