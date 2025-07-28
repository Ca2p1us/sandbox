import os
import json

def log(file_path: str, answer):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # ファイルが存在しない場合は新規作成
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump({"results": [answer]}, f, indent=2)
        return

    with open(file_path, 'r') as f:
        content = f.read()
        if not content.strip():
            read_data = []
        else:
            read_data = json.loads(content).get("results", [])
        save_data = read_data + [answer]

    with open(file_path, 'w') as f:
        json.dump({"results": save_data}, f, indent=2)

    return