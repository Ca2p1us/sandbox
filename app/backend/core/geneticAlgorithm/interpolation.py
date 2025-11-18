#事前評価・適応度評価を行う

from typing import List, Dict
import math
from scipy.interpolate import RBFInterpolator
import numpy as np

RNG = np.random.default_rng(seed=10)

def interpolate_by_distance(
    population: List[dict],
    best: dict = None,
    worst: dict = None,
    param_keys: List[str] = None,
    target_key: str = "pre_evaluation"
):
    """
    best, worstを基準に、個体群のtarget_key（pre_evaluationやfitnessなど）を距離に応じて線形補間で再評価する。
    best/worstがNoneの場合はランダムな値を付与する。
    param_keys: 距離計算に使うパラメータ名リスト（Noneならoperator1のfrequencyのみ）
    target_key: 補間して格納するキー名（"pre_evaluation"または"fitness"など）
    """
    print(f"補間を開始します。")
    if not best or not worst:
        print(f"bestまたはworstがNoneです。ランダムな{target_key}を付与します。")
        for ind in population:
            ind[target_key] = RNG.uniform(1.0, 10.0)
        return

    if param_keys is None:
        # デフォルトはoperator1のfrequencyのみ
        param_keys = ["fmParamsList.operator1.frequency"]

    def get_param(ind, key):
        # "fmParamsList.operator1.frequency" のようなドット区切りでアクセス
        val = ind
        for k in key.split('.'):
            val = val.get(k, None)
            if val is None:
                break
        return val

    # ベクトル化
    def to_vec(ind):
        return [get_param(ind, k) for k in param_keys]

    best_vec = to_vec(best)
    worst_vec = to_vec(worst)

    # 距離計算
    def euclidean(vec1, vec2):
        return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(vec1, vec2) if a is not None and b is not None))

    max_dist = euclidean(best_vec, worst_vec)
    if max_dist == 0:
        # 全員同じ場合
        for ind in population:
            value = best.get(target_key, best.get("fitness", 1))
            ind[target_key] = float(value)
        print(f"距離が同一です。\nbest {best[target_key]}, worst {worst[target_key]}")
        return

    best_val = float(best.get("fitness", 1))
    worst_val = float(worst.get("fitness", 1))

    for ind in population:
        ind_vec = to_vec(ind)
        dist_best = euclidean(ind_vec, best_vec)
        # 距離が近いほどbest_valに近づく
        ratio = dist_best / max_dist
        value = best_val * (1 - ratio) + worst_val * ratio
        ind[target_key] = value
    print(f"{target_key}を補間しました。\nbest {best_val}, worst {worst_val}")
    return


def interpolate_by_Gaussian(
    population: List[dict],
    best: dict = None,
    worst: dict = None,
    param_keys: List[str] = None,
    target_key: str = "pre_evaluation",
    eps_ratio: float = 0.02,
    eps_floor_ratio: float = 1e-6
):
    """
    ガウス関数 f(x) = A exp(-(x-μ)^2/(2σ^2)) + C のパラメータ推定
    C ≈ y2 として、わずかに ε を下げた近似で計算。
    
    eps_ratio: 全振幅に対する補正割合（例: 0.02 = 2%）
    """
    print(f"ガウス補間を開始します。")
    if not best or not worst:
        print(f"bestまたはworstがNoneです。ランダムな{target_key}を付与します。")
        for ind in population:
            ind[target_key] = RNG.uniform(1.0, 10.0)
        return
    if param_keys is None:
        # デフォルトはoperator1のfrequencyのみ
        param_keys = ["fmParamsList.operator1.frequency"]
    def get_param(ind, key):
        # "fmParamsList.operator1.frequency" のようなドット区切りでアクセス
        val = ind
        for k in key.split('.'):
            val = val.get(k, None)
            if val is None:
                break
        return val
    best_params = [get_param(best, k) for k in param_keys]
    worst_params = [get_param(worst, k) for k in param_keys]

    if any(p is None for p in best_params) or any(p is None for p in worst_params):
        raise ValueError("best/worst に param_keys が存在しません。")
    
    best_val = float(best.get("fitness", 1.0))
    worst_val = float(worst.get("fitness", 1.0))
    eps = (best_val - worst_val) * eps_ratio
    C = worst_val - eps
    A = best_val - C
    print(f"best_val: {best_val}, worst_val: {worst_val}")
    print(best[target_key] - C)

    # ratio for sigma estimation（数値安定化）
    denom = (best.get("fitness", 1e-12) - C)
    if denom == 0:
        raise ValueError("best の target が C と等しい。sigma を推定できません。")
    ratio = (worst.get("fitness", 1e-12) - C) / denom
    if not (0 < ratio < 1):
        raise ValueError("ratio が (0,1) の範囲にない。best/worst の target を確認してください。")

    sigma = []
    for b, w in zip(best_params, worst_params):
        raw = abs(b - w)
        # if raw == 0, we still need a meaningful scale; use relative floor
        floor = max(abs(b), abs(w), 1.0) * eps_floor_ratio
        raw = max(raw, floor)
        s = raw / math.sqrt(-2.0 * math.log(ratio))
        s = max(s, floor)  # avoid exact zero
        sigma.append(s)
    mu = best_params
    print(f"ガウス補間のパラメータ: \n\tA={A}, \n\tmu={mu}, \n\tsigma={sigma}, \n\tC={C}")

    # 個体群のガウス補間
    for ind in population:
        ind_params = [get_param(ind, k) for k in param_keys]
        if any(p is None for p in ind_params):
            ind[target_key] = RNG.uniform(1.0, 10.0)
            continue
        # compute normalized squared distance
        dist_sq = 0.0
        for xj, muj, sj in zip(ind_params, mu, sigma):
            z = (xj - muj) / sj
            dist_sq += z * z
        # note: DO NOT divide by N; the scale is already handled by sigma
        value = A * math.exp(-0.5 * dist_sq) + C
        ind[target_key] = value
    print(f"{target_key}をガウス補間しました。\nbest {best_val}, worst {worst_val}")
    return

def interpolate_by_RBF(
    evaluate_population: List[str],      # 評価済み個体のUUIDリスト
    population: dict,        # {UUID: 個体オブジェクト} の辞書
    population_to_interp: List[dict], # 補間対象の個体オブジェクトリスト
    param_keys: List[str] = None,
    target_key: str = "pre_evaluation",
    N_SAMPLES: int = 60
):
    """
    UUIDリストで管理された評価済み個体（上位N_SAMPLES）の fitness を用いて、
    population_to_interp の pre_evaluation をRBF補間する。
    カーネルはGaussianを使用する。
    """
    print(f"RBF補間（Gaussian, N={N_SAMPLES}）を開始します。")

    FITNESS_KEY = "fitness"
    
    def get_param(ind, key):
        val = ind
        for k in key.split('.'):
            val = val.get(k, None)
            if val is None:
                break
        return val

    def to_vec(ind):
        vec = []
        for k in param_keys:
            val = get_param(ind, k)
            try:
                # None または非数値の場合は 0.0 を使用
                vec.append(float(val) if val is not None else 0.0)
            except ValueError:
                vec.append(0.0)
        return vec
    # --- ヘルパー関数ここまで ---

    if param_keys is None:
        param_keys = ["fmParamsList.operator1.frequency"]
    
    # --- 1. サンプリングデータの準備（上位60個体の抽出） ---
    
    # UUIDリストから実際の個体オブジェクトを取得
    evaluated_objects = []
    for uid in evaluate_population:
        ind = population.get(uid)
        # 個体が存在し、かつ fitness を持っていることを確認
        if ind and ind.get(FITNESS_KEY) is not None:
             evaluated_objects.append(ind)
    
    if len(evaluated_objects) < N_SAMPLES:
        print(f"警告: 評価済み個体数が不足しています (現在: {len(evaluated_objects)})。補間をスキップします。")
        # フォールバック処理 (population_to_interp 全体にランダム値を付与)
        for ind in population_to_interp:
            ind[target_key] = RNG.uniform(1.0, 10.0)
        return

    # fitness の降順でソートし、上位 N_SAMPLES 個体を選択
    # reverse=True で値が大きい（良い）順
    rbf_samples = sorted(
        evaluated_objects,
        key=lambda ind: ind.get(FITNESS_KEY, -np.inf),
        reverse=True
    )[:N_SAMPLES]

    # パラメータベクトル X_train と評価値 Y_train を抽出
    X_train = np.array([to_vec(ind) for ind in rbf_samples])
    Y_train = np.array([ind.get(FITNESS_KEY, 0.0) for ind in rbf_samples], dtype=float)

    if X_train.size == 0 or Y_train.size == 0 or np.any(np.isnan(X_train)) or np.any(np.isnan(Y_train)):
        print("エラー: サンプルデータが不正です。RBF補間をスキップします。")
        return

    # --- 2. RBFInterpolator モデルの作成と補間 ---
    
    # kernel='gaussian' を指定し、epsilon='auto' で最適な滑らかさを自動推定
    rbfi = RBFInterpolator(X_train, Y_train, kernel='gaussian', epsilon='auto')

    # 補間対象の個体のパラメータベクトル X_predict を作成
    X_predict = np.array([to_vec(ind) for ind in population_to_interp])
    
    # 補間実行
    try:
        Y_predict = rbfi(X_predict)
    except Exception as e:
        print(f"RBF補間中にエラーが発生しました: {e}。補間をスキップします。")
        return
    
    # 補間結果を個体群に格納
    for ind, value in zip(population_to_interp, Y_predict):
        # 予測値が浮動小数点数であることを保証
        ind[target_key] = float(value)
        
    print(f"{target_key}をRBF補間しました。N_SAMPLES={N_SAMPLES}, kernel=Gaussian")
    return