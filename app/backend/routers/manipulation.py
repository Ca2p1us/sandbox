import uuid
from typing import Optional
from utils.geneticAlgorithm.config import ALGORITHM_NUM, GENERATION_CHROMOSOME_NUM
from schemas.chromosome import  ChromosomesParams
from utils.geneticAlgorithm.gene_repair.fm_params.repair import repair_fm_params
from utils.geneticAlgorithm.mutate.mutate_fm_params.mutate import mutate
from utils.geneticAlgorithm.crossover.blx_alpha import exec_blx_alpha
from utils.geneticAlgorithm.selection.elite_selection import exec_elite_selection
from utils.geneticAlgorithm.selection.tournament_selection import exec_tournament_selection
from utils.geneticAlgorithm.make_chromosome_params.four_oscillator.make_chromosome_params import make_chromosome_params
# from utils.geneticAlgorithm.make_chromosome_params.three_oscilator.test_int_overtones import make_chromosome_params
# from utils.geneticAlgorithm.make_chromosome_params.three_oscilator.test_int_overtones import make_chromosome_params
from utils.log.log import log
from utils.validation.validation import validation
from utils.sort_modulator import sort_modulators

from fastapi import APIRouter, Cookie

router = APIRouter()


@router.get("/manipulation/initialize")
async def make_generation_chromosomes():
    """すべての染色体を初期化する

    Returns:
        [type]: 染色体を返す
    """
    return {
        "chromosome1": make_chromosome_params(),
        "chromosome2": make_chromosome_params(),
        "chromosome3": make_chromosome_params(),
        "chromosome4": make_chromosome_params(),
        "chromosome5": make_chromosome_params(),
        "chromosome6": make_chromosome_params(),
        "chromosome7": make_chromosome_params(),
        "chromosome8": make_chromosome_params(),
        "chromosome9": make_chromosome_params(),
        "chromosome10": make_chromosome_params()
    }


@router.post("/manipulation")
async def gene_manipulation(chromosomes_params: ChromosomesParams, giongo: Optional[str] = Cookie(None), random_strings: Optional[str] = Cookie(None)) -> ChromosomesParams:
    chromosomes_params: dict = chromosomes_params.dict()
    # 被験者名
    name = chromosomes_params.pop('name')
    age = chromosomes_params.pop('age')
    gender = chromosomes_params.pop('gender')
    hearing = chromosomes_params.pop('hearing')

    # # 入力のバリデーション
    validation(name,age,gender,hearing,chromosomes_params)
    # 回答を記録
    # 呼び出し元ファイルからの相対パスを渡す（今回はbackend）
    log_dirPath = "./results/" + f"{giongo}/"
    log_file_name = name+"-"+age+"-"+gender+"-"+hearing+"-" + random_strings + ".json"
    log_path = log_dirPath + log_file_name
    log(log_path, chromosomes_params)
    # 次世代の染色体を入れる配列
    next_generation_chromosomes: list[dict] = []
    # 適応度が最も高い個体を保存（エリート保存）
    elite_chromosome = exec_elite_selection(chromosomes_params)
    # エリート個体の適応度を初期化して次世代用配列に格納
    elite_chromosome["fitness"] = ""
    next_generation_chromosomes.append(elite_chromosome)
    # モジュレーターをソート(FMのアルゴリズム上ソートが必要な場合がある)
    for chromosome in chromosomes_params.values():
        modulators_params = [chromosome["fmParamsList"]["operator2"],
                             chromosome["fmParamsList"]["operator3"], chromosome["fmParamsList"]["operator4"]]
        sorted_modulators_params = sort_modulators(modulators_params)
        chromosome["fmParamsList"]["operator2"] = sorted_modulators_params[0]
        chromosome["fmParamsList"]["operator3"] = sorted_modulators_params[1]
        chromosome["fmParamsList"]["operator4"] = sorted_modulators_params[2]
    # 交叉による次世代個体の追加
    for i in range(len(next_generation_chromosomes), GENERATION_CHROMOSOME_NUM):
        parents = exec_tournament_selection(chromosomes_params)
        offspring = exec_blx_alpha(parents, repair_fm_params, mutate)
        offspring["fitness"] = ""
        offspring["chromosomeId"] = uuid.uuid4()
        offspring["algorithmNum"] = ALGORITHM_NUM
        next_generation_chromosomes.append(offspring)
    return {
        "chromosome1": next_generation_chromosomes[0],
        "chromosome2": next_generation_chromosomes[1],
        "chromosome3": next_generation_chromosomes[2],
        "chromosome4": next_generation_chromosomes[3],
        "chromosome5": next_generation_chromosomes[4],
        "chromosome6": next_generation_chromosomes[5],
        "chromosome7": next_generation_chromosomes[6],
        "chromosome8": next_generation_chromosomes[7],
        "chromosome9": next_generation_chromosomes[8],
        "chromosome10": next_generation_chromosomes[9]
    }
