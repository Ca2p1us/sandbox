import random
from typing import List

# 完全ランダムな評価（1〜10）
def evaluate_fitness_random(population: List[dict]):
    for group in population:
        for individual in group:
            if not isinstance(individual, dict):
                print("individualの型:", type(individual), "値:", individual)
                continue  # またはraise Exceptionで止めてもOK
            individual["fitness"] = str(random.randint(1, 10))

# FMパラメータに基づいた評価（例：operator1のfrequencyに基づく）
def evaluate_fitness_by_param(population: List[dict]):
    for group in population:
        for individual in group:
            op1 = individual["fmParamsList"]["operator1"]
            frequency = op1.get("frequency", 0)
            # 例えば frequency を使って評価（220Hz ～ 880Hz を1～10にマッピング）
            normalized = max(1, min(10, int((frequency - 220) / (880 - 220) * 9 + 1)))
            individual["fitness"] = str(normalized)
