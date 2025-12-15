import json
import os
from pathlib import Path
def combine_json_files(input_folder, output_file):
    combined_data = {}

    # 入力フォルダ内のすべてのJSONファイルを取得
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                combined_data.update(data)  # データを結合

    # 結合したデータを出力ファイルに保存
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)  # 出力フォル
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)
if __name__ == "__main__":
    Gauss_input_folders = ['result/proposal/best/Gaussian/Gauss/200inds_9eval',
                     'result/conventional/best/Gaussian/200inds',
                     'result/conventional/best/Gaussian/9inds']
    # input_folder = 'result/proposal/best/Gaussian/Gauss/200inds_9eval'  # 入力フォルダのパス
    # input_folder = 'result/conventional/best/Gaussian/200inds'
    # input_folder = 'result/conventional/best/Gaussian/9inds'
    Gauss_output_files = ['result/proposal/best/Gaussian/Gauss/combined/combined_fitness_log_200inds_9eval.json',
                    'result/conventional/best/Gaussian/combined/combined_fitness_log_200inds.json',
                    'result/conventional/best/Gaussian/combined/combined_fitness_log_9inds.json']
    # output_file = 'result/proposal/best/Gaussian/Gauss/combined/combined_fitness_log.json'  # 出力ファイルのパス
    # output_file = 'result/conventional/best/Gaussian/combined/combined_fitness_log_200inds.json'
    # output_file = 'result/conventional/best/Gaussian/combined/combined_fitness_log_9inds.json'
    IDW_input_folders = ['result/proposal/best/Gaussian/IDW/200inds_9eval',
                     'result/conventional/best/Gaussian/200inds',
                     'result/conventional/best/Gaussian/9inds']
    IDW_output_files = ['result/proposal/best/Gaussian/IDW/combined/combined_fitness_log_200inds_9eval.json',
                    'result/conventional/best/Gaussian/combined/combined_fitness_log_200inds.json',
                    'result/conventional/best/Gaussian/combined/combined_fitness_log_9inds.json']
    for input_folder, output_file in zip(IDW_input_folders, IDW_output_files):
        combine_json_files(input_folder, output_file)
        print(f"Combined JSON files from {input_folder} into {output_file}")