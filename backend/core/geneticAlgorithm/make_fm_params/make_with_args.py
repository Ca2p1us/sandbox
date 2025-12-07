ALGORITHM_NUM = 0

def make_fm_params_with_args(
        attack:float, #アタックの時間
        decay:float, #ディケイの時間
        sustain:float, #サステインのレベル
        sustain_time:float, #サステインの時間
        release:float, #リリースの時間
        frequency:float, #周波数(kHz)
        ):

    """FMのパラメータを返す

    Returns:
        dict: [description]
    """
    return {
        "attack": attack,
        "decay": decay,
        "sustain": sustain,
        "sustainTime": sustain_time,
        "release": release,
        "frequency": frequency
    }
