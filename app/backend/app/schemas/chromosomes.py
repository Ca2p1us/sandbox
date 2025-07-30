# モデルの定義
from typing import Any, Dict
from pydantic import BaseModel


# class ChromosomesParams(BaseModel):
# dict


class ChromosomeParams(BaseModel):
    algorithmNum: int
    fitness: int
    pre_evaluation: int = 0  # 事前評価値（デフォルト0、型や初期値は用途に応じて変更）
    chromosomeId: str
    operator1: dict
    operator2: dict
    operator3: dict
    operator4: dict


class ChromosomesParams(BaseModel):
    chromosome1: dict
    chromosome2: dict
    chromosome3: dict
    chromosome4: dict
    chromosome5: dict
    chromosome6: dict
    chromosome7: dict
    chromosome8: dict
    chromosome9: dict
    chromosome10: dict
    # 被験者情報
    name: str
    age: str
    gender: str
    hearing: str
