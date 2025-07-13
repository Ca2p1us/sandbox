import json
import glob
import csv

def create_analyze_file():
    gions:list = ["pi","bu-","fa-n","ti-n"]
    one_gion_results_paths:list = glob.glob(f"./fixed_evaluation_results/pi/*.json")
    one_gion_result_file_names = [file_name.split("/")[-1] for file_name in one_gion_results_paths]
    all_result:list = []
    # 誰が音を生成したか記録
    sound_generator = [""]
    for personal_file_name in one_gion_result_file_names:
        personal_result = [personal_file_name]
        for gion in gions:
            personal_result_path = f"./fixed_evaluation_results/{gion}/{personal_file_name}"
            with open(personal_result_path, 'r') as f:
                read_data:dict = json.load(f)
                for sound_path, each_sound_result in read_data.items():
                    personal_result.extend(each_sound_result["fitness"])
                    splited_sound_path = sound_path.split("/")[-1]
                    splited_sound_path = splited_sound_path.split("-")
                    if len(sound_generator) <= 48:
                        sound_generator.append(splited_sound_path[0]+".json")
        all_result.append(personal_result)
    all_result.insert(0,sound_generator)
    with open("./analyzation/fitness.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows(all_result)

