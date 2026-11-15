import numpy as np

from dqisim import registry
from dqisim.evaluators import Context


def instance(name="xorsat-regular", seed=0, **kw):
    return registry.build("problem", {"name": name, **kw}).generate(np.random.default_rng(seed))


def ctx(inst, ell, decoder="bp", weights="optimal-ideal", postselect=True, seed=1):
    dec = decoder if isinstance(decoder, dict) else {"name": decoder}
    w = weights if isinstance(weights, dict) else __import__("dqisim.spec", fromlist=["x"]).parse_component(weights)
    return Context(inst, np.random.default_rng(seed), ell, registry.build("weights", w), dec, postselect)
