import uuid
import random
import numpy as np
from ..geneticAlgorithm import config

from ..geneticAlgorithm.config import ALGORITHM_NUM, IGNORE_RANGE
from ..geneticAlgorithm.make_fm_params.make_with_config import make_fm_params
from ..geneticAlgorithm.make_fm_params.make_with_args import make_fm_params_with_args
from ..geneticAlgorithm.make_fm_params.make_zero_params import make_zero_params


# 各パラメータの定義域（例: min, max値）
OPERATOR_PARAM_RANGES = {
    "attack": config.ATTACK_RANGE,
    "decay": config.DECAY_RANGE,
    "sustain": config.SUSTAIN_RANGE,
    "sustainTime": config.SUSTAIN_TIME_RANGE,
    "release": config.RELEASE_RANGE,
    "frequency": config.FREQUENCY_RANGE,  # kHz単位
}

# 1. シード値固定のためのGeneratorオブジェクトを作成
# シード値 (例: 42) を指定することで、乱数のシーケンスが固定されます。
RNG = np.random.default_rng(seed=242)

def generate_random_fm_params_list(param_ranges: dict) -> list[float]:
    """
    OPERATOR_PARAM_RANGESに基づいて、シード値固定の乱数パラメータのリストを生成する。
    make_fm_params_with_argsの引数順に値を返す。
    """
    params = {}
    
    # 辞書のキーの順番を固定し、make_fm_params_with_argsの引数順に乱数を生成
    param_order = ["attack", "decay", "sustain", "sustainTime", "release", "frequency"]
    
    for key in param_order:
        low, high = param_ranges[key]
        # NumPyのGenerator.uniform(low, high)は low <= x < high の浮動小数点数を生成
        params[key] = RNG.uniform(low, high)
        # while IGNORE_RANGE[0] <= params[key] <= IGNORE_RANGE[1]:
        #     params[key] = RNG.uniform(low, high)
        
    # make_fm_params_with_args の引数として展開できるように、順番にリスト化して返す
    return [params[key] for key in param_order]

def make_chromosome_params():
    """ランダムな値を持つ個体を生成する(実験に使用)

    Returns:
        [type]: [description]
    """
    #乱数のリストを生成
    fm_params_list = generate_random_fm_params_list(OPERATOR_PARAM_RANGES)
    random_fm_params = make_fm_params_with_args(*fm_params_list)

    return {
        "fmParamsList": {
            "operator1": random_fm_params
            # "operator2": make_fm_params_with_args(0,0,1,3,0,2200)
        },
        # "algorithmNum": ALGORITHM_NUM,
        "fitness": 0.0,
        "pre_evaluation": 0.0,  # 事前評価値（デフォルト0、型や初期値は用途に応じて変更）
        "generation": 1,  # 世代数（デフォルト0、用途に応じて変更）
        "chromosomeId": uuid.uuid4()
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
if __name__ == "__main__":
    # --- 動作確認 ---
    # 1回目の呼び出し
    result1 = make_chromosome_params()["fmParamsList"]["operator1"]
    print("--- 1回目 ---")
    print(result1)

    # 2回目の呼び出し（同じRNGオブジェクトを使用し続けるため、異なる乱数が出る）
    result2 = make_chromosome_params()["fmParamsList"]["operator1"]
    print("--- 2回目（RNG継続）---")
    print(result2)

    # Generatorを同じシードで再初期化すると、同じ乱数シーケンスが再現される
    RNG = np.random.default_rng(42) 
    result3 = make_chromosome_params()["fmParamsList"]["operator1"]
    print("--- 3回目（RNG再初期化）---")
    print(result3)

    # 1回目と3回目の結果は一致します（シード固定の再現性）
    print(result1 == result3) # True