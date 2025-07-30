import uuid

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
            "operator1": make_fm_params_with_args(0,0,1,3,0,440,1,300),
            "operator2": make_fm_params_with_args(0,0,1,3,0,2200,1,1500),
            # "operator2": make_zero_params(),
            "operator3": make_zero_params(),
            # "operator3": make_fm_params_with_args(0,0,1,3,0,88,1,50),
            "operator4": make_zero_params(),
        },
        "algorithmNum": ALGORITHM_NUM,
        "fitness": "",
        "pre_evaluation": 0,  # 事前評価値（デフォルト0、型や初期値は用途に応じて変更）
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
