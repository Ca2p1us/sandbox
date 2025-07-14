import math
import random

from ...config import ALGORITHM_NUM
from ...make_fm_params.make_odd_overtones import make_fm_params
from ...make_fm_params.make_zero_params import make_zero_params


def make_chromosome_params():
    """奇数倍音のみが発生することを確認

    Returns:
        [type]: [description]
    """
    return {
        "fmParamsList": {
            "operator1": make_fm_params(),
            "operator2": make_fm_params(),
            "operator3": make_zero_params(),
            "operator4": make_zero_params(),
        },
        "algorithmNum": 0,
        "fitness": ""
    }
