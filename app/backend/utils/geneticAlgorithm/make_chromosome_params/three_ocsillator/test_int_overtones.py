import math
import random

from ...make_fm_params.make_with_args import make_fm_params 

def make_chromosome_params():
    """整数倍音のみが発生することを確認

    Returns:
        [type]: [description]
    """


    return {
        "fmParamsList": {
            "operator1": make_fm_params(0,0,1,1,0,500,2,100),
            "operator2": make_fm_params(0,0,1,1,0,500,2,0),
            "operator3": make_fm_params(0,0,1,1,0,500,2,0),
            "operator4": make_fm_params(0,0,1,1,0,500,1,100),
        },
        "algorithmNum": 1,
        "fitness": ""
    }
