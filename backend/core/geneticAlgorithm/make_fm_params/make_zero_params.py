def make_zero_params():
    """作用しないオシレーターのパラメータを生成

    Returns:
        [type]: [description]
    """
    return {
        "attack": 0,
        "decay": 0,
        "sustain": 1,
        "sustainTime": 1,
        "release": 0,
        "frequency": 440,
        "ratioToFundamentalFrequency": 1,
        "modulationIndex": 0
    }
