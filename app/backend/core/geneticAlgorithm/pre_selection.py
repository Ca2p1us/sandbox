#評価する個体を選ぶプログラム

from typing import List

def select_top_individuals_by_fitness(population: List[dict], top_n: int = 10) -> List[dict]:
    """
    fitness（事前評価済み）で降順ソートし、上位top_n個体を返す
    """
    # fitnessが未設定または変換できない個体は除外
    valid_population = [ind for ind in population if "fitness" in ind and str(ind["fitness"]).isdigit()]
    # 降順ソート
    sorted_population = sorted(valid_population, key=lambda x: int(x["fitness"]), reverse=True)
    return sorted_population[:top_n]