import json
import glob

def create_fit_chromosome_json(gion:str,save_path:str):
    result_paths:list = glob.glob(f"./results/{gion}/*.json")
    best_fit_chromosomes:dict = {}
    for path in result_paths:
        with open(path, 'r') as f:
            read_data = json.load(f)
            best_fit_chromosome = max(
            read_data[12].values(), key=lambda params: params["fitness"])

            best_fit_chromosomes[path] = best_fit_chromosome 

    with open(save_path, 'w') as f:
        json.dump(best_fit_chromosomes, f)

