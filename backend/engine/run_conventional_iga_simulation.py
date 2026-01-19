
# demo_rgb.py
import numpy as np
from saf_ieda import SAFIEDA, SAFIEDAConfig

def rgb_distance_score(pop_rgb: np.ndarray, target=(255, 255, 255)) -> np.ndarray:
    """
    pop_rgb: [Nc, 3] 各値は 0..255
    目標色とのユークリッド距離を 1..99 に線形マッピング
    """
    tgt = np.array(target)[None, :]
    dist = np.linalg.norm(pop_rgb - tgt, axis=1)
    # 最大距離は原点(0,0,0) から (255,255,255) で約 441.67
    maxd = np.sqrt(3*(255**2))
    score = 99.0 * (1.0 - dist / maxd)
    return np.clip(score, 1.0, 99.0)

def make_eval_callback_rgb():
    def callback(top_pop_attr_ids: np.ndarray):
        """
        変数 c=3, 属性 m=256 とみなし、属性ID=そのまま色値。
        """
        Nc = top_pop_attr_ids.shape[0]
        # 属性ID (0..255) をそのまま R,G,B
        rgb = top_pop_attr_ids.astype(float)
        scores = rgb_distance_score(rgb, target=(255, 255, 255))  # 合成
        times = np.ones(Nc)  # 評価時間(秒)を 1 と仮置き
        meta = {}
        return scores, times, meta
    return callback

if __name__ == "__main__":
    cfg = SAFIEDAConfig(N=200, c=3, m=256, Nc=12, ef_max=99.0, lambda_=2.3)
    algo = SAFIEDA(cfg)

    cb = make_eval_callback_rgb()

    def stopper(g, info):
        # 代理: Top-Nc 平均スコアが 98 を超えたら終了
        return info["scores"].mean() >= 98.0

    hist = algo.run(max_gen=50, eval_callback=cb, stop_callback=stopper)
    print(f"Finished at gen={hist[-1]['gen']}, avg score={hist[-1]['scores'].mean():.2f}")
