import os
import json
from typing import Optional
from typing import Dict, Any
from schemas.chromosome import  ChromosomesParams
from utils.create_fit_chromosome_json import create_fit_chromosome_json
from utils.validation.validation import validation
from utils.analyze.analyze_evaluation import create_analyze_file

from fastapi import APIRouter, Cookie

router = APIRouter()


# 分析用のファイルを作成
@router.get("/evaluation/all/create_analyze")
async def create_file_for_analyze():
    create_analyze_file()
    return {"result":"分析用ファイルが作成されました"}

# 聴取実験用のファイルを作成
@router.get("/evaluation/{gion}/create")
async def create_json_for_evaluate(gion:str):
    create_fit_chromosome_json(gion,f"./best_fit_chromosomes/{gion}/best_fit.json")
    return {"result":"jsonファイルが作成されました"}


# 聴取実験の結果を記録
@router.post("/evaluation/log")
async def log_chromosomes_fitness(chromosomes_params: dict, giongo: Optional[str] = Cookie(None), random_strings: Optional[str] = Cookie(None)) -> ChromosomesParams:
    # 被験者名
    name = chromosomes_params.pop('name')
    age = chromosomes_params.pop('age')
    gender = chromosomes_params.pop('gender')
    hearing = chromosomes_params.pop('hearing')

    # # 入力のバリデーション
    validation(name,age,gender,hearing,chromosomes_params)
    # 回答を記録
    # 呼び出し元ファイルからの相対パスを渡す（今回はbackend）
    log_dir_path = "./evaluation_results/" + f"{giongo}/"
    log_file_name = name+"-"+age+"-"+gender+"-"+hearing+"-" + random_strings + ".json"
    log_path = log_dir_path + log_file_name
    # 実験の結果を記録
    if os.path.exists(log_path) == False:
        with open(log_path, 'w') as f:
            json.dump(chromosomes_params, f)
    return {"greeting":"実験へのご協力ありがとうございます。"}

@router.get("/evaluation/{gion}/initialize")
async def initialize_chromosomes(gion:str) -> Dict[str, Any]:
    evaluation_chromosomes_file_path :str= f"./best_fit_chromosomes/{gion}/best_fit.json"
    with open(evaluation_chromosomes_file_path, 'r') as f:
        read_data = json.load(f)
        return read_data

