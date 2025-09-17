# seeds.py
import hashlib
import os
import random

import numpy as np
from faker import Faker


def derive_seed(base_seed: int, *scope) -> int:
    s = str(base_seed) + "::" + "::".join(map(str, scope))
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


class SeedBundle:
    def __init__(self, base_seed: int, *scope):
        self.base_seed = base_seed
        self.scope = scope
        self.py = random.Random(derive_seed(base_seed, *scope, "py"))
        self.np = np.random.default_rng(derive_seed(base_seed, *scope, "np"))
        self.faker = Faker()
        self.faker.seed_instance(derive_seed(base_seed, *scope, "faker"))


def seed_environment(base_seed: int):
    # stable hashing -> stable set/dict iteration
    os.environ["PYTHONHASHSEED"] = str(base_seed)
    random.seed(base_seed)
    np.random.seed(base_seed)  # optional; prefer local RNGs below
    Faker.seed(base_seed)
