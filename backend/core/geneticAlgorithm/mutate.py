import random
from ..geneticAlgorithm.config import *
import numpy as np

RNG = np.random.default_rng(seed=40)

def mutate(param_name: str):
    if param_name == "attack":
        fm_param = choice_from_all_range(
            ATTACK_RANGE[0], ATTACK_RANGE[1])
        return fm_param
    if param_name == "decay":
        fm_param = choice_from_all_range(DECAY_RANGE[0], DECAY_RANGE[1])
        return fm_param
    if param_name == "sustain":
        fm_param = choice_from_all_range(
            SUSTAIN_RANGE[0], SUSTAIN_RANGE[1])
        return fm_param
    if param_name == "sustainTime":
        fm_param = choice_from_all_range(
            SUSTAIN_TIME_RANGE[0], SUSTAIN_TIME_RANGE[1])
        return fm_param
    if param_name == "release":
        fm_param = choice_from_all_range(
            RELEASE_RANGE[0], RELEASE_RANGE[1])
        return fm_param
    if param_name == "frequency":
        fm_param = choice_from_all_range(
            FREQUENCY_RANGE[0], FREQUENCY_RANGE[1])
        return fm_param
    if param_name == "ratioToFundamentalFrequency":
        fm_param = choice_from_all_range(
            RATIO_TO_CARRIER_FREQUENCY_RANGE[0], RATIO_TO_CARRIER_FREQUENCY_RANGE[1])
        return fm_param
    if param_name == "modulationIndex":
        fm_param = choice_from_all_range(
            MODULATION_INDEX_RANGE[0], MODULATION_INDEX_RANGE[1])
        return fm_param

    return None


def choice_from_all_range(min,max):
    return RNG.uniform(min,max)