import uuid
import random

from ..geneticAlgorithm.config import ALGORITHM_NUM
from ..geneticAlgorithm.make_fm_params.make_with_config import make_fm_params
from ..geneticAlgorithm.make_fm_params.make_with_args import make_fm_params_with_args
from ..geneticAlgorithm.make_fm_params.make_zero_params import make_zero_params

def make_chromosome_params():
    """ランダムな値を持つ個体を生成する(実験に使用)

    Returns:
        [type]: [description]
    """
    return {
        "fmParamsList": {
            "operator1": make_fm_params_with_args(0.1,0.1,0.5,0.2,0.1,0.440)
            # "operator2": make_fm_params_with_args(0,0,1,3,0,2200)
        },
        "algorithmNum": ALGORITHM_NUM,
        "fitness": 0.0,
        "pre_evaluation": 0.0,  # 事前評価値（デフォルト0、型や初期値は用途に応じて変更）
        "chromosomeId": uuid.uuid4()
    }

# def make_chromosome_params():
#     """ランダムな値を持つ個体を生成する(実験に使用)
#     Returns:
#         [type]: [description]
#     """
#     return {
#         "fmParamsList": {
#             "operator1": make_fm_params(),
#             "operator2": make_fm_params(),
#             "operator3": make_fm_params(),
#             "operator4": make_fm_params(),
#         },
#         "algorithmNum": ALGORITHM_NUM,
#         "fitness": "",
#         "chromosomeId": uuid.uuid4()
#     }

# 各パラメータの定義域（例: min, max値）
OPERATOR_PARAM_RANGES = {
    "attack": (0.0, 0.25),
    "decay": (0.0, 0.25),
    "sustain": (0.0, 1.0),
    "sustainTime": (0.0, 0.3),
    "release": (0.0, 0.35),
    "frequency": (0.200, 1.300),  # kHz単位
}

def make_operator_random():
    """1オペレータ分のパラメータを定義域内でランダム生成"""
    return {
        key: random.uniform(*OPERATOR_PARAM_RANGES[key])
        for key in OPERATOR_PARAM_RANGES
    }

def make_chromosome_random():
    """4オペレータ分のパラメータをランダム生成した個体を返す"""
    return {
        "fmParamsList": {
            "operator1": make_operator_random(),
            "operator2": make_operator_random(),
            "operator3": make_operator_random(),
            "operator4": make_operator_random(),
        },
        "algorithmNum": ALGORITHM_NUM,
        "fitness": "",
        "pre_evaluation": 0,
        "chromosomeId": uuid.uuid4()
    }
