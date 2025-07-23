
from ..core.geneticAlgorithm.selection import tournament
from ..core.geneticAlgorithm.make_chromosome_params import *
from ..core.geneticAlgorithm import *
from ..engine import population_utils

NUM_GENERATIONS = 100
POPULATION_SIZE = 50

def run_simulation():
    # 1. 初期個体生成
    population = population_utils.initialize_population()
    
    for generation in range(NUM_GENERATIONS):
        # 2. 評価
        fitnesses = make_chromosome_params(population)

        # 3. 選択
        selected = tournament.exec_tournament_selection(population, fitnesses)

        # 4. 交叉 & 突然変異
        offspring = reproduce(selected)

        # 5. 次世代への更新
        population = update_population(population, offspring, fitnesses)

    # 6. 最終結果の出力
    return population
