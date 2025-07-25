
from ..core.geneticAlgorithm.selection import tournament
from ..core.geneticAlgorithm.make_chromosome_params import *
from ..core.geneticAlgorithm import BLX_alpha
from ..engine import population_utils
from ..core.geneticAlgorithm import make_chromosome_params
from ..engine import evaluate

NUM_GENERATIONS = 10
POPULATION_SIZE = 10

def make_initial_population(num_individuals=10):
    return [make_chromosome_params() for _ in range(num_individuals)]

def run_simulation():
    # 1. 初期個体生成
    population = make_initial_population(POPULATION_SIZE)
    
    for generation in range(NUM_GENERATIONS):
        # 2. 評価
        fitnesses = evaluate(population)

        # 3. 選択
        selected = tournament.exec_tournament_selection(population)

        # 4-1. 交叉
        offspring = BLX_alpha.exec_blx_alpha(
            parents_chromosomes=selected,
            func_repair_gene=population_utils.repair_gene,
            mutate=population_utils.mutate
        )

        # 5. 次世代への更新
        population = offspring

    # 6. 最終結果の出力
    return population
