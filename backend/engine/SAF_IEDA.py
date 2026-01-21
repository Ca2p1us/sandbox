import numpy as np
from scipy.optimize import minimize

# --- SAF-IEDA Helper Class ---

class SAF_SurrogateModel:
    """
    SAF-IEDA (Surrogate-Assisted Fitness Interactive Estimation of Distribution Algorithm)
    のための適応度予測モデル
    """
    def __init__(self, num_vars, lamb=2.3):
        self.c = num_vars
        self.lamb = lamb
        self.weights = np.ones(self.c) / self.c
        self.top_individuals = None
        self.top_fitness = None
        self.f_max = 1.0

    def fit(self, top_individuals, top_fitness):
        """
        Top-Nc個体を用いて決定変数の重み(weights)を最適化する
        top_individuals: np.array shape (Nc, c)
        top_fitness: np.array shape (Nc,)
        """
        self.top_individuals = top_individuals
        self.top_fitness = top_fitness
        self.f_max = np.max(top_fitness)
        Nc = len(top_individuals)

        # 目的関数: Leave-one-out 交差検証的な予測誤差の最小化
        def objective(w):
            error = 0.0
            # 重みの正規化 (合計が1になるように)
            w_sum = np.sum(w)
            norm_w = w / (w_sum + 1e-9)
            
            for k in range(Nc):
                actual_f = top_fitness[k]
                # 自分自身(k)を除いた個体群を参照データとする
                mask = np.arange(Nc) != k
                train_set = top_individuals[mask]
                train_fit = top_fitness[mask]
                
                # 自分自身の適応度を予測
                pred_f = self._predict_single(top_individuals[k], norm_w, train_set, train_fit)
                error += abs(pred_f - actual_f) ** self.lamb
            
            return error ** (1.0 / self.lamb)

        # 最適化実行 (SLSQP法)
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = [(0.01, 1.0) for _ in range(self.c)]
        init_w = np.ones(self.c) / self.c
        
        # エラーハンドリング: Ncが小さすぎる場合は最適化をスキップして等分配
        if Nc < 2:
            self.weights = init_w
        else:
            try:
                res = minimize(objective, init_w, bounds=bounds, constraints=constraints, method='SLSQP')
                self.weights = res.x / np.sum(res.x)
            except Exception as e:
                print(f"Warning: Optimization failed ({e}), using uniform weights.")
                self.weights = init_w

    def predict(self, individuals):
        """
        個体群の適応度を予測する
        individuals: np.array shape (N, c)
        return: np.array shape (N,)
        """
        if self.top_individuals is None:
            return np.zeros(len(individuals))
            
        predictions = []
        for ind in individuals:
            pred = self._predict_single(ind, self.weights, self.top_individuals, self.top_fitness)
            predictions.append(pred)
        return np.array(predictions)

    def _predict_single(self, target_vec, weights, ref_pop, ref_fit):
        """
        1個体の予測計算 (SAF-IEDA Eq. 1 & 5)
        """
        c = self.c
        mus = np.zeros(c)
        denom = np.sum(ref_fit)
        
        if denom == 0: return 0.0

        # 次元ごとの類似度計算
        # ここでは簡易的に「パラメータ定義域の 10%〜20%」程度をシグマとする
        # もし全変数が同じスケール(0-255など)なら、そのスケールを変数として渡すのがベスト
        # 仮に config から範囲が取れない場合、ref_pop の分散から推定する
        # ref_pop (Top-Nc) の標準偏差の平均などをベースにする
        pop_std = np.std(ref_pop, axis=0)
        avg_std = np.mean(pop_std)
        
        # 分散が0になってしまった場合の対策
        if avg_std < 1e-6:
            sigma = 1.0 # デフォルト値
        else:
            # Top-Ncの広がりの「半分」程度を類似判定のカーネル幅にするなど
            sigma = avg_std * 2.0
        
        for i in range(c):
            # i次元目の値の距離
            dists = np.abs(target_vec[i] - ref_pop[:, i])
            # ガウスカーネルによる類似度
            sims = ref_fit * np.exp(-(dists**2) / (2 * sigma**2))
            mus[i] = np.sum(sims) / denom

        # 重み付き和で適応度を算出 (Eq. 5)
        return np.sum(weights * mus) * self.f_max