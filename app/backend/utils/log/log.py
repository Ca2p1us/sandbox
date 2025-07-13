import os
import json

def log(file_path: str, answer):
    if os.path.exists(file_path) == False:
        with open(file_path, 'w') as f:
            json.dump([answer], f)
        return
    
    with open(file_path, 'r') as f:
        read_data = json.load(f)
        save_data = read_data + [answer]

    with open(file_path, 'w') as f:
        json.dump(save_data, f)

    return