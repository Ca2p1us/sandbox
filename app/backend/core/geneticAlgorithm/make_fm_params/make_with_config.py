import random
from ..config import *
from ..round_off import *

ALGORITHM_CANDIDATE = [1]

# エンベロープの候補
ATTACK_CANDIDATE = [0, 0.1, 0.2, 0.3, 0.4]
DECAY_CANDIDATE = [0]
SUSTAIN_CANDIDATE = [1]
RELEASE_CANDIDATE = [0.1, 0.2, 0.3, 0.4, 0.5]


def make_fm_params():
    return {
        "attack": random.uniform(ATTACK_RANGE[0], ATTACK_RANGE[1]),
        "decay": random.uniform(DECAY_RANGE[0], DECAY_RANGE[1]),
        "sustain": random.uniform(SUSTAIN_RANGE[0], SUSTAIN_RANGE[1]),
        "sustainTime": random.uniform(SUSTAIN_TIME_RANGE[0], SUSTAIN_TIME_RANGE[1]),
        "release": random.uniform(RELEASE_RANGE[0], RELEASE_RANGE[1]),
        "frequency": random.uniform(FREQUENCY_RANGE[0], FREQUENCY_RANGE[1]),
        "ratioToFundamentalFrequency": random.uniform(RATIO_TO_CARRIER_FREQUENCY_RANGE[0], RATIO_TO_CARRIER_FREQUENCY_RANGE[1]),
        "modulationIndex": random.uniform(MODULATION_INDEX_RANGE[0], MODULATION_INDEX_RANGE[1])
    }

# 操作のピッ
# 警報のブー
# 終了のチーンはこのパラメータで出来そう
# 呼び出しのピリリリなどはできない
