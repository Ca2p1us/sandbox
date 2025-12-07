import matplotlib.pyplot as plt
import numpy as np
import os

def plot_gauss_function():
    # 1. パラメータの設定
    mu = 250        # 平均 (μ)
    sigma = 75      # 標準偏差 (σ)
    x_min, x_max = 0, 500  # 定義域 (x軸の範囲)
    y_min, y_max = 0, 6    # 値域 (y軸の範囲)
    
    # ガウス関数のピーク（高さ）の設定
    # 注意: 通常の確率密度関数のピークは約0.008と低いため、
    # 指定された値域[0, 6]で見えるようにピークを6.0に設定します。
    amplitude = 6.0 

    # 2. データの生成
    x = np.linspace(x_min, x_max, 1000)
    
    # ガウス関数の計算: f(x) = A * exp( - (x - μ)^2 / (2σ^2) )
    y = amplitude * np.exp(-((x - mu)**2) / (2 * sigma**2))

    # 4. プロット設定
    fig, ax = plt.subplots(figsize=(8, 6)) # fig, ax を取得するように変更
    
    # 4. 指定された点の計算 (x=250, x=0)
    points_x = [250, 0]
    points_y = [amplitude * np.exp(-((px - mu)**2) / (2 * sigma**2)) for px in points_x]
    
    # ガウス関数の描画
    ax.plot(x, y, color='black', linewidth=1.5, label='Gaussian Curve')
    
    # 点のプロット (赤色、サイズ大きめ)
    ax.scatter(points_x, points_y, color='blue', s=100, zorder=5, label='Points')
    
    # 点の座標を表示
    
    ax.text(points_x[0], points_y[0] + 0.1, f'best', ha='center', fontsize=12)
    ax.text(points_x[1]+20, points_y[1] + 0.2, f'worst', ha='center', fontsize=12)

    # 軸（枠線）の太さを変える
    axis_width = 1.5  # 太さの設定
    for spine in ax.spines.values():
        spine.set_linewidth(axis_width)

    # メモリ（数字）のフォントサイズと、メモリ自体の太さを変える
    ax.tick_params(axis='both', which='major', labelsize=16, width=axis_width, length=6)

    # x軸: 0, 100, 200, 300, 400, 500 (np.arangeは終了値を含まないので+1します)
    ax.set_xticks(np.arange(0, 501, 50)) 
    
    # y軸: 0, 1, 2, 3, 4, 5, 6
    ax.set_yticks(np.arange(0, 7, 1))
    
    # 軸の範囲設定
    ax.set_xlim(x_min-0.5, x_max+0.5)
    ax.set_ylim(y_min, y_max+0.5)
    
    # グリッドを消す
    ax.grid(False)
    
    # ラベル等の設定
    ax.set_title("")
    ax.set_xlabel('parameter value',fontsize=18)
    ax.set_ylabel('fitness',fontsize=18)
    
    # 4. グラフの保存と表示
    save_dir = "for_slide"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    filename = os.path.join(save_dir, "gaussian_plot_styled.png")
    plt.savefig(filename,bbox_inches='tight')
    print(f"グラフを {filename} として保存しました。")
    
    plt.show()


if __name__ == "__main__":
    plot_gauss_function()