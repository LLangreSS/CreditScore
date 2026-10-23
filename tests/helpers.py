from functools import lru_cache

from creditrisk import data
from creditrisk.model import train_scoring_model


@lru_cache(maxsize=1)
def demo_clean():
    clean, _ = data.clean_and_engineer(data.generate_demo_data(n=12_000))
    return clean


@lru_cache(maxsize=1)
def trained():
    return train_scoring_model(demo_clean(), backend="hgb")
