import json
import os
def combine_json_files(input_folder, output_file):
    combined_data = []

    # 入力フォルダ内のすべてのJSONファイルを取得
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                combined_data.extend(data)  # データを結合

    # 結合したデータを出力ファイルに保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)
if __name__ == "__main__":
    input_folder = 'backend/engine/result/proposal/fitness_logs'  # 入力フォルダのパス
    output_file = 'backend/engine/result/proposal/combined_fitness_log.json'  # 出力ファイルのパス
    combine_json_files(input_folder, output_file)
    print(f"Combined JSON files from {input_folder} into {output_file}")