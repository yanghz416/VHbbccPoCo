from pocket_coffea.lib.weights.weights import WeightLambda

import numpy as np

def weight_vjets(events, params, metadata):
    dr  = np.array(events.dijet.deltaR)
    wei = np.ones_like(dr)

    wei[(dr>0.01) & (dr<1.0)] = 1.3

    return wei

custom_weight_vjet  = WeightLambda.wrap_func(
    name="weight_vjet",
    function=lambda params, metadata, events, size, shape_variations:
    weight_vjets(events, params, metadata),
    has_variations=True
)
