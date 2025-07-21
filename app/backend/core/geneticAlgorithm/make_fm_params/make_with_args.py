ALGORITHM_NUM = 0

def make_fm_params_with_args(
        attack:float,
        decay:float,
        sustain:float,
        sustain_time:float,
        release:float,
        frequency:float,
        ratio_to_fundamental_frequency:int,
        modulation_index:float
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
        "frequency": frequency,
        "ratioToFundamentalFrequency": ratio_to_fundamental_frequency,
        "modulationIndex": modulation_index
    }
