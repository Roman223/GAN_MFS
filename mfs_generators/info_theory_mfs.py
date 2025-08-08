import torch
from typing import Optional, Union

from pymfe.mfe import MFE
import pandas as pd
import numpy as np

class StatisticalMFS:
    device = torch.device("cpu")

    @staticmethod
    def ft_