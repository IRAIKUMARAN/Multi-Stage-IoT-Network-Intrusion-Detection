"""Shared helpers: IO, metrics, plotting, seeding."""


def set_seed(seed: int = 42): # to keep the shuffing same as everytime
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
