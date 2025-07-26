import uuid
import random
from typing import Dict
from pydantic import BaseModel

ALGORITHM_NUM = 0  # 固定値


class ChromosomeParams(BaseModel):
    algorithmNum: int
    fitness: int
    chromosomeId: str
    operator1: Dict
    operator2: Dict
    operator3: Dict
    operator4: Dict


class ChromosomesParams(BaseModel):
    chromosome1: Dict
    chromosome2: Dict
    chromosome3: Dict
    chromosome4: Dict
    chromosome5: Dict
    chromosome6: Dict
    chromosome7: Dict
    chromosome8: Dict
    chromosome9: Dict
    chromosome10: Dict
    name: str
    age: str
    gender: str
    hearing: str


def make_fm_params_with_args(
    attack: float,
    decay: float,
    sustain: float,
    sustain_time: float,
    release: float,
    frequency: float,
    ratio_to_fundamental_frequency: int,
    modulation_index: float
) -> Dict:
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


def create_random_operator() -> Dict:
    return make_fm_params_with_args(
        attack=random.uniform(0.01, 1.0),
        decay=random.uniform(0.01, 1.0),
        sustain=random.uniform(0.0, 1.0),
        sustain_time=random.uniform(0.1, 2.0),
        release=random.uniform(0.01, 1.0),
        frequency=random.uniform(100.0, 1000.0),
        ratio_to_fundamental_frequency=random.randint(1, 10),
        modulation_index=random.uniform(0.1, 10.0)
    )


def create_chromosome() -> ChromosomeParams:
    return ChromosomeParams(
        algorithmNum=ALGORITHM_NUM,
        fitness=0,
        chromosomeId=str(uuid.uuid4()),
        operator1=create_random_operator(),
        operator2=create_random_operator(),
        operator3=create_random_operator(),
        operator4=create_random_operator()
    )


def initialize_population(name: str, age: str, gender: str, hearing: str) -> ChromosomesParams:
    return ChromosomesParams(
        chromosome1=create_chromosome().dict(),
        chromosome2=create_chromosome().dict(),
        chromosome3=create_chromosome().dict(),
        chromosome4=create_chromosome().dict(),
        chromosome5=create_chromosome().dict(),
        chromosome6=create_chromosome().dict(),
        chromosome7=create_chromosome().dict(),
        chromosome8=create_chromosome().dict(),
        chromosome9=create_chromosome().dict(),
        chromosome10=create_chromosome().dict(),
        name=name,
        age=age,
        gender=gender,
        hearing=hearing
    )
