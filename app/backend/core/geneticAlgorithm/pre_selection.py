#評価する個体を選ぶプログラム

from typing import List

def select_top_individuals_by_pre_evaluation(population: List[dict], top_n: int = 10) -> List[str]:
    """
    pre_evaluation（事前評価済み）で降順ソートし、上位top_n個体のIDリストを返す
    """
    # pre_evaluationが未設定または変換できない個体は除外
    valid_population = [ind for ind in population if "pre_evaluation" in ind and str(ind["pre_evaluation"]).isdigit() and "chromosomeId" in ind]
    # 降順ソート
    sorted_population = sorted(valid_population, key=lambda x: int(x["pre_evaluation"]), reverse=True)
    # 個体IDリストで返す
    return [ind["chromosomeId"] for ind in sorted_population[:top_n]]