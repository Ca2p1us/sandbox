import matplotlib.pyplot as plt
import matplotlib.animation as ani
import os
import json
import numpy as np
import pyaudio
import wave
from scipy.signal import sawtooth
import uuid
from .geneticAlgorithm.config import ATTACK_RANGE, NUM_GENERATIONS
from typing import List, Optional, Union
import japanize_matplotlib
from pathlib import Path

plt.rcParams['font.family'] = 'MS Gothic'

# --- 定数・マッピング定義 ---

# 評価関数 ID -> 名前
EVALUATE_MAP = {
    1: "Gaussian",
    2: "Sphere",
    3: "Gaussian_cos",
    4: "Ackley",
    5: "Gaussian_peaks"
}
# 名前 -> 評価関数 ID (逆引き用)
EVALUATE_NAME_TO_ID = {v: k for k, v in EVALUATE_MAP.items()}

# 補間手法 ID -> 名前
INTERPOLATE_MAP = {
    0: "linear",
    1: "Gauss",
    2: "RBF",
    3: "IDW",
    4: "Hybrid",
    100: None,  # 補間なし等の特別扱い
    None: None
}
# 名前 -> 補間手法 ID (逆引き用、None除く)
INTERPOLATE_NAME_TO_ID = {v: k for k, v in INTERPOLATE_MAP.items() if v is not None}

# グラフのY軸範囲設定
# 関数ごとに微妙に異なっていた値を統合的に管理します。
# 必要に応じてここを調整してください。
Y_LIM_SETTINGS = {
    "Gaussian": (0, 6.5),
    "Ackley": (0, 6.0),
    "Gaussian_peaks": (0, 6.5),
    "Gaussian_cos": (0, 9.5), # 以前のコードで最大範囲だったものを採用
    "Sphere": (-100000, 0.5)
}

# --- ヘルパー関数 ---

def _get_method_name(evaluate_input: Union[int, str]) -> str:
    """IDまたは名前から評価関数名を取得 (堅牢化)"""
    # 整数IDの場合
    if isinstance(evaluate_input, int):
        return EVALUATE_MAP.get(evaluate_input, "Unknown")
    # 文字列の場合 (例: "Ackley")
    elif isinstance(evaluate_input, str):
        if evaluate_input in EVALUATE_NAME_TO_ID:
            return evaluate_input
        # IDが文字列で渡された場合 ("4"など)
        if evaluate_input.isdigit():
             return EVALUATE_MAP.get(int(evaluate_input), "Unknown")
    
    return "Unknown"

def _get_interpolate_name(interpolate_input: Union[int, str, None]) -> Optional[str]:
    """IDまたは名前から補間手法名を取得 (堅牢化)"""
    # Noneの場合
    if interpolate_input is None:
        return None
        
    # 整数IDの場合
    if isinstance(interpolate_input, int):
        return INTERPOLATE_MAP.get(interpolate_input, "Unknown")
        
    # 文字列の場合
    elif isinstance(interpolate_input, str):
        if interpolate_input in INTERPOLATE_NAME_TO_ID:
            return interpolate_input
        # "None" という文字列の場合
        if interpolate_input == "None":
            return None
        if interpolate_input.isdigit():
             return INTERPOLATE_MAP.get(int(interpolate_input), "Unknown")

    return "Unknown"

def _setup_plot(ax, method: str, x_label='Generation', y_label='Fitness', title=''):
    """グラフの共通設定を適用するヘルパー関数"""
    ax.set_xlabel(x_label, fontsize=18)
    ax.set_ylabel(y_label, fontsize=18)
    ax.set_title(title)
    ax.set_xlim(0.5, NUM_GENERATIONS + 0.5)
    
    if method in Y_LIM_SETTINGS:
        ax.set_ylim(*Y_LIM_SETTINGS[method])
    
    ax.grid(True)

def _get_save_path(ver: str, method: str, interpolate: Optional[str], category: str, file_name: str) -> Path:
    """保存パスを生成する共通関数"""
    
    
    # 1. 比較グラフ (result/comparison/...)
    if ver == "comparison":
        if interpolate is None:
             base_dir = Path(f'./result/comparison/{method}')
        else:
             base_dir = Path(f'./result/comparison/{method}/{interpolate}')
        
    # 2. 平均適応度 (result/{ver}/average/...)
    elif category == "average":
        if interpolate is None:
            base_dir = Path(f'./result/{ver}/average/{method}')
        else:
            base_dir = Path(f'./result/{ver}/average/{method}/{interpolate}')

    # 3. その他グラフ (result/{ver}/graph/...)
    else:
        # category: "best_fitnesses", "error_histories"
        if interpolate is None:
            base_dir = Path(f'./result/{ver}/graph/{method}/{category}')
        else:
            base_dir = Path(f'./result/{ver}/graph/{method}/{interpolate}/{category}')
            
    
    # ディレクトリの作成 (ここが重要: 確実にフォルダを作る)
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"ディレクトリ作成エラー: {e}")

    # ファイル名の結合
    # file_name が既にパス区切り文字を含んでいると意図しない挙動になるため、
    # method名とfile_nameの結合をここで行う
    full_path = base_dir / f"{method}{file_name}"
    
    # print(f"保存先パス: {full_path}") # デバッグ用出力
    return full_path


def log(file_path: str, answer,times: int = 1):
    # UUID型をstr型に変換するヘルパー関数
    def convert_uuid_to_str(obj):
        if isinstance(obj, dict):
            return {k: convert_uuid_to_str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_uuid_to_str(v) for v in obj]
        elif isinstance(obj, uuid.UUID):
            return str(obj)
        else:
            return obj

    save_data = convert_uuid_to_str(answer)
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open(mode="w", encoding="utf-8") as f:
        json.dump({f"{times}_results": save_data}, f, indent=2)

    return

def log_error_history(evaluate_num: int = None, interpolate_num: int = None, file_path: str = None, error_history: list = None, ver: str = None):
    """
    世代ごとのbest個体のfitness履歴をグラフ表示する
    best_fitness_history: [(世代番号, fitness値), ...] のリスト
    """
    if not error_history:
        print("履歴データがありません。")
        return
    
    method = _get_method_name(evaluate_num)
    interpolate = _get_interpolate_name(interpolate_num)

    save_path = _get_save_path(ver = ver, method=method, interpolate=interpolate, category="error_histories", file_name=file_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots()

    generations = [item[0] for item in error_history]
    error_values = [item[1] for item in error_history]

    ax.set_xlabel('Generation')  # x軸ラベル
    ax.set_ylabel('Error')  # y軸ラベル
    ax.set_title("")  # グラフタイトル
    ax.set_xlim(0.5,NUM_GENERATIONS+0.5)
    ax.grid(True)

    ax.plot(generations, error_values,marker='o', linestyle='-', color='blue', label='Error')
    ax.legend(loc=0)

    plt.savefig(save_path)
    plt.close()


def log_fitness(evaluate_num: int = None, interpolate_num: int = None, file_path: str = None, best_fitness_history: list = None, average_fitness_history: list = None, ver: str = None):
    """
    世代ごとのbest個体のfitness履歴をグラフ表示する
    best_fitness_history: [(世代番号, fitness値), ...] のリスト
    """
    if not best_fitness_history:
        print("履歴データがありません。")
        return
    
    method = _get_method_name(evaluate_num)
    interpolate = _get_interpolate_name(interpolate_num)

    save_path = _get_save_path(ver, method, interpolate, "best_fitnesses", file_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots()

    generations = [item[0] for item in best_fitness_history]
    fitness_values = [item[1] for item in best_fitness_history]
    average_values = [item[1] for item in average_fitness_history] if average_fitness_history else None

    _setup_plot(ax,method,y_label='Fitness')

    ax.plot(generations, fitness_values,
             marker='o', linestyle='-', color='blue', label='Best Fitness')
    if average_values:
        ax.plot(generations, average_values,
                 marker='o', linestyle='--', color='orange', label='Average Fitness')
    ax.legend(loc=0)
    plt.savefig(save_path)
    # plt.show()
    plt.close()

def log_fitness_histories(evaluate_num: int = None, interpolate_num: int = None, file_path: str = None, best_fitness_histories = None, ver: str = None):
    """
    複数回のシミュレーションの世代ごとのbest個体のfitness履歴をグラフ表示する
    best_fitness_histories: [[(世代番号, fitness値), ...], ...] のリスト
    """
    if not best_fitness_histories:
        print("履歴データがありません。")
        return

    method = _get_method_name(evaluate_num)
    interpolate = _get_interpolate_name(interpolate_num)

    save_path = _get_save_path(ver, method, interpolate, "best_fitnesses", file_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, best_fitness_history in enumerate(best_fitness_histories):
        generations = [item[0] for item in best_fitness_history]
        fitness_values = [item[1] for item in best_fitness_history]
        ax.plot(generations, fitness_values,
                 marker='o', linestyle='-', label=f'Run {idx+1}')

    _setup_plot(ax,method,y_label='Best Fitness', title=f'{method} Best Fitness Histories')
    plt.savefig(save_path)
    # plt.show()
    plt.close()
    return

def log_average_fitness(evaluate_num: int = None, interpolate_num:int = None,file_path: str = None, average_fitness_history=None, times: int = None):
    """
    世代ごとのaverage個体のfitness履歴をjsonファイルに保存する
    average_fitness_history: [(世代番号, fitness値), ...] のリスト
    """
    method = _get_method_name(evaluate_num)
    interpolate = _get_interpolate_name(interpolate_num)

    # verの決定: 補間ありならproposal, なしならconventional
    ver = "proposal" if interpolate is not None else "conventional"
    
    save_path = _get_save_path(ver, method, interpolate, "average", file_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # JSONファイルに保存
    with save_path.open('w', encoding='utf-8') as f:
        json.dump({f"{times}_average_fitness":average_fitness_history}, f, indent=2)

    return
def log_comparison(evaluate_num: int = None, interpolate_num: int = None, file_path: str = None, plot_series_list: list = None , indicator : str = None):
    """
    複数回のシミュレーションの世代ごとのbest個体のfitness履歴をグラフ表示する
    best_fitness_histories: [[(世代番号, fitness値), ...], ...] のリスト
    Args:
        evaluate_num (int): 評価関数のID
        interpolate_num (int): 補間手法のID
        file_path_suffix (str): 保存ファイル名のサフィックス (例: "_comparison.png")
        plot_series_list (list): 描画するデータのリスト。
            形式: [
                {'label': '凡例名1', 'data': [(世代, fit), ...], 'marker': 'o', 'linestyle': '-'},
                {'label': '凡例名2', 'data': [(世代, fit), ...], 'marker': 'x', 'linestyle': '--'},
                ...
            ]
        indicator (str): Y軸ラベルの接頭辞
    """

    if not plot_series_list:
        print("データがありません。")
        return
    method = _get_method_name(evaluate_num)
    interpolate = _get_interpolate_name(interpolate_num)

    # 比較用パス生成 (ver="comparison" とする)
    save_path = _get_save_path("comparison", method, interpolate, "", file_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)


    fig, ax = plt.subplots(figsize=(8, 5))

    # リスト内のデータをループで描画
    max_gen = 0
    for series in plot_series_list:
        data = series.get('data', [])
        if not data:
            continue
            
        generations = [item[0] for item in data]
        fitness_values = [item[1] for item in data]
        
        # 軸の最大値更新用
        if generations:
            max_gen = max(max_gen, max(generations))

        ax.plot(
            generations, 
            fitness_values,
            marker=series.get('marker', 'o'),  # 指定がなければ 'o'
            linestyle=series.get('linestyle', '-'), # 指定がなければ '-'
            label=series.get('label', 'No Label')
        )
    
    _setup_plot(ax,method,y_label=indicator+'Fitness')
    # X軸範囲設定 (データに合わせて動的に設定、または定数NUM_GENERATIONSを使用)
    ax.set_xlim(0.5, max_gen + 0.5 if max_gen > 0 else NUM_GENERATIONS + 0.5)
    ax.legend(prop={"family": "MS Gothic", "size": 14}, loc=0)
    # fig.tight_layout()
    # plt.savefig(f'./result/graph/{method}_fitness_histories.png')
    plt.savefig(save_path)
    plt.show()
    plt.close()
    return

def log_compare(evaluate_num: int = None, interpolate_num: int = None, file_path: str = None, fitness_histories: list = None, tornament_sizes: list = None, population_size: int = None):
    """
    複数のトーナメントサイズのシミュレーション結果を1枚の画像にまとめる関数
    param_keys: 少なくとも6つのキーが必要
    """
    method = _get_method_name(evaluate_num)
    interpolate = _get_interpolate_name(interpolate_num)

    # タイトル設定
    title_text = f"{population_size}個体 " + ("補間あり" if interpolate else "補間なし")
    
    save_path = _get_save_path("comparison", method, interpolate, "", file_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))


    for history in fitness_histories:
        if not history.any():
            print("評価データがありません。")
            return
        generations = [item[0] for item in history]
        values = [item[1] for item in history]
        ax.plot(generations, values, marker='o', linestyle='-', label='Fitness')
    
    _setup_plot(ax,method,y_label='Fitness', title=title_text)

    ax.legend([f"k={size}" for size in tornament_sizes],prop={"family":"MS Gothic","size":14},loc=0)
    plt.savefig(save_path)
    plt.show()
    plt.close()
    return

# サンプリング周波数と再生時間の設定
SAMPLE_RATE = 44100  # CD品質のサンプリングレート
CARRIER_FREQ_BASE = 440.0  # 基本となるキャリア周波数（A4音）


def load_params_from_json(file_path):
    """
    JSONファイルからパラメータを読み込む関数
    """
    if not os.path.exists(file_path):
        print(f"エラー: ファイルが見つかりません -> {file_path}")
        return None

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def generate_adsr_envelope(params, duration, sample_rate):
    """
    ADSRパラメータに基づいてエンベロープ波形を生成する関数 (前の回答と同じ)
    """
    attack_time = params['attack']
    decay_time = params['decay']
    sustain_level = params['sustain']
    sustain_time = params['sustainTime']
    release_time = params['release']

    attack_samples = int(attack_time * sample_rate)
    decay_samples = int(decay_time * sample_rate)
    sustain_samples = int(sustain_time * sample_rate)
    release_samples = int(release_time * sample_rate)
    total_samples = int(duration * sample_rate)

    envelope = np.zeros(total_samples)

    # Attack, Decay, Sustain, Releaseの各セクションの処理
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    if decay_samples > 0:
        envelope[attack_samples:attack_samples +
                 decay_samples] = np.linspace(1, sustain_level, decay_samples)
    envelope[attack_samples + decay_samples:attack_samples +
             decay_samples + sustain_samples] = sustain_level
    release_start_idx = attack_samples + decay_samples + sustain_samples
    if release_samples > 0:
        envelope[release_start_idx:release_start_idx +
                 release_samples] = np.linspace(sustain_level, 0, release_samples)

    return envelope[:total_samples]


def play_sound(waveform, sample_rate):
    """
    指定された波形をPyAudioで再生する関数 (前の回答と同じ)
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=sample_rate,
                    output=True)

    audio_data = (waveform * 0.5).astype(np.float32).tobytes()
    stream.write(audio_data)

    stream.stop_stream()
    stream.close()
    p.terminate()

# WAVファイルに書き出す関数


def save_sound_as_wave(waveform, sample_rate, filename):
    """
    指定された波形をWAVファイルに保存する関数
    """
    # 波形データを16ビットPCMに変換
    waveform_integers = np.int16(waveform * 32767)

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # モノラル
        wf.setsampwidth(2)  # 16ビット
        wf.setframerate(sample_rate)
        wf.writeframes(waveform_integers.tobytes())

    print(f"WAVファイルを保存しました: {filename}")

# --- メイン処理 ---


def sound_check(file_path=None):
    method = file_path[18:]
    method = method[:-5]
    # 1. JSONファイルからデータを読み込む
    fm_data = load_params_from_json(file_path)

    if fm_data:
        # 最も評価値の高い個体を取得、複数いた場合は最初の個体を選択
        best_individual = max(
            fm_data["results"][-1],
            key=lambda ind: float(ind.get("fitness", 0))
        )
        params = best_individual['fmParamsList']['operator1']

        # 2. ADSRエンベロープと波形を生成
        NOTE_DURATION = params['attack'] + params['decay'] + \
            params['sustainTime'] + params['release']
        envelope = generate_adsr_envelope(params, NOTE_DURATION, SAMPLE_RATE)
        t = np.linspace(0, NOTE_DURATION, len(envelope), endpoint=False)
        frequency = params['frequency'] * 1000  # 周波数をHzに変換
        # キャリア波形の生成（ここではサイン波を使用）
        carrier_wave = np.sin(2 * np.pi * frequency * t)

        # 3. エンベロープを適用
        modulated_wave = carrier_wave * envelope

        # 4. 音声の再生
        print("JSONファイルからパラメータを読み込み、音声を再生します...")
        play_sound(modulated_wave, SAMPLE_RATE)
        print("再生が終了しました。")
        # 5. WAVファイルに保存
        save_sound_as_wave(modulated_wave, SAMPLE_RATE,
                           f"./result/sound/{method}.wav")
        print(f"WAVファイルに保存しました: app/result/sound/{method}.wav")
    else:
        print("音声の再生をスキップします。")

def plot_individual_params(population: list[dict],best: dict,worst: dict, param_keys: list[str], generation: int, file_path: str):
    """
    個体群の指定されたパラメータを3つのペアでプロットし、1枚の画像にまとめる関数
    param_keys: 少なくとも6つのキーが必要
    """
    def get_param(ind, key):
        # "fmParamsList.operator1.frequency" のようなドット区切りでアクセス
        val = ind
        for k in key.split('.'):
            val = val.get(k, None)
            if val is None:
                break
        return val

    # 1行3列のサブプロットを作成 (figsizeは横長に設定)
    
    # プロットするペアのインデックスリスト
    pair_indices = [(0, 1), (2, 3), (4, 5)]


    # 3つのグラフをループで描画
    for i, (idx1, idx2) in enumerate(pair_indices):
        fig, ax = plt.subplots(figsize=(6, 6))
        key1 = param_keys[idx1]
        key2 = param_keys[idx2]

        # データの抽出
        param_values1 = [get_param(ind, key1) for ind in population]
        param_values2 = [get_param(ind, key2) for ind in population]
        best_value1 = get_param(best, key1)
        best_value2 = get_param(best, key2)
        worst_value1 = get_param(worst, key1)
        worst_value2 = get_param(worst, key2)

        # プロット設定
        ax.scatter(param_values1, param_values2, alpha=1.0) # 重なりが見やすいように透過度を設定
        ax.scatter(best_value1, best_value2, color='red', s=100, label='Best', edgecolors='black') # 最良個体を強調表示
        ax.scatter(worst_value1, worst_value2, color='blue', s=100, label='Worst', edgecolors='black') # 最悪個体を強調表示
        ax.legend(['個体群', 'best', 'worst',],prop={"family":"MS Gothic"})
        
        # 軸範囲の設定 (元のコードに準拠)
        ax.set_xlim(-50, ATTACK_RANGE[1] + 50)
        ax.set_ylim(-50, ATTACK_RANGE[1] + 50)
        
        # ラベルとグリッド
        ax.set_xlabel(f"param{idx1+1}", fontsize=9) # パラメータ番号をラベルにする
        ax.set_ylabel(f"param{idx2+1}", fontsize=9)
        ax.set_title(f"{generation}世代({idx1+1},{idx2+1}次元目)")
        ax.grid(True)

        # レイアウトを自動調整して重なりを防ぐ
        plt.tight_layout()
        
        # ファイル保存
        file_path_save = Path(f"{file_path}_pair{i+1}.png")
        file_path_save.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(file_path_save)
        plt.close() # メモリ解放
    return


