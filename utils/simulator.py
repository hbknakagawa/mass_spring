import numpy as np

class MassSpringSimulator:
    def __init__(self, m=1.0, k=10.0, c=1.0, l=1.0, dt=0.001, T=30.0):
        self.m = m
        self.k = k
        self.c = c
        self.l = l
        self.dt = dt
        self.T = T
        self.t = np.arange(0, T, dt)  # これなら t[0] = 0.0 が保証される
        self.N = len(self.t)
        # self.N = int(T / dt)                # ステップ数
        # self.t = np.linspace(0, T, self.N)  # 時刻配列

    def drive_signal(self, mode='exp', params=None):
        """
        駆動点(x)の運動パターンを生成する関数
        exp, const, linear, sine, random, triangle, step, trapezoid
        """
        if params is None:
            params = {}

        if mode == 'exp':
            A = params.get('A', self.l)
            b = params.get('b', 1.0)
            return A * (1 - np.exp(-b * self.t))
        elif mode == 'const':
            A = params.get('A', self.l)
            return np.clip(self.t / self.T, 0, 1) * A  # [0, 1]に収束
        elif mode == 'linear':
            v = params.get('v', 0.1)
            return np.clip(v * self.t / self.T, 0, 1)  # [0, 1]に収束
        elif mode == 'sine':
            A = params.get('A', self.l)
            b = params.get('b', 1.0)
            return 0.5 * (np.sin(b * self.t) + 1)  # [0, 1] の範囲に収束
        elif mode == 'random':
            A = params.get('A', self.l)
            return np.clip(np.random.random(self.t.shape), 0, 1)  # [0, 1] に収束
        elif mode == 'triangle':
            A = params.get('A', self.l)
            period = params.get('period', 1.0)
            return A * (2 * np.abs(np.mod(self.t, period) / period - 0.5) - 1)
        elif mode == 'step':
            A = params.get('A', self.l)
            step_time = params.get('step_time', 5.0)
            return np.where(self.t >= step_time, A, 0.0)  # ステップ状に変化
        elif mode == 'trapezoid':
            A = params.get('A', self.l)
            rise_time = params.get('rise_time', 20.0)
            return np.clip(self.t / rise_time, 0, 1) * A  # 台形プロファイル
        elif mode == 'wavy_decay':
            A = params.get('A', 0.3)
            b = params.get('b', 0.3)
            w = params.get('w', 5.0)
            decay = np.exp(-b * self.t)
            wavy = A * np.sin(w * self.t)       # 初期値は 0
            return (1 - decay) + decay * wavy   # 初期値 0、最終的に 1 に収束
        else:
            raise ValueError(f"Unsupported mode: {mode}")    

    def simulate(self, drive_type='exp', drive_param=None):
        """
        バネ－質点システムの数値シミュレーション（オイラー法）
        - drive_type: 駆動モード
        - drive_param: 駆動モードに応じたパラメータ
        """
        g = 9.81  # 重力加速度 [m/s^2]
        # 駆動点の位置配列 x(t) を生成
        x = self.drive_signal(mode=drive_type, params=drive_param)
        # 質点の位置、速度、加速度配列
        p = np.zeros(self.N)
        v = np.zeros(self.N)
        a = np.zeros(self.N)

        # 初期条件：釣り合い位置に設定
        x0 = x[0]
        p[0] = x0 - self.l - (self.m * g / self.k)
        v[0] = 0.0

        # オイラー法で逐次更新
        for i in range(self.N - 1):
            stretch = (p[i] - x[i]) + self.l
            a[i] = (-self.k * stretch - self.c * v[i]) / self.m - g
            v[i + 1] = v[i] + a[i] * self.dt
            p[i + 1] = p[i] + v[i + 1] * self.dt

        # 最終ステップの加速度
        stretch = (p[-1] - x[-1]) + self.l
        a[-1] = (-self.k * stretch - self.c * v[-1]) / self.m - g

        vx = np.gradient(x, self.dt, edge_order=2)
        ax = np.gradient(vx, self.dt, edge_order=2)

        return self.t, x, vx, ax, p, v, a

    
    def simulate_without_g(self, drive_type='exp', drive_param=None):
        """
        バネ－質点システムの数値シミュレーション(オイラー法)
        - drive_type: 駆動モード(exp, const, linea)
        - drive_param: 駆動モードに応じたパラメータ
        """
        # 駆動点の位置配列 x(t) を生成
        x = self.drive_signal(mode=drive_type, params=drive_param)
        # 質点の位置、速度、加速度配列
        p = np.zeros(self.N)
        v = np.zeros(self.N)
        a = np.zeros(self.N)

        # 初期条件
        p[0] = -self.l  # バネの自然長だけ離れた位置
        v[0] = 0.0

        # オイラー法で逐次更新
        for i in range(self.N - 1):
            # バネの伸び：現在のバネの長さ - 自然長
            stretch = (p[i] - x[i]) + self.l

            # 運動方程式（F=ma）で加速度を計算
            a[i] = (-self.k * stretch - self.c * v[i]) / self.m

            # 速度・位置の更新（Euler法）
            v[i + 1] = v[i] + a[i] * self.dt
            p[i + 1] = p[i] + v[i + 1] * self.dt

        # 最終ステップの加速度を計算（補完）
        a[-1] = (-self.k * ((p[-1] - x[-1]) + self.l) - self.c * v[-1]) / self.m

        # 駆動点の速度 vx(t) と加速度 ax(t) は数値微分で求める
        vx = np.gradient(x, self.dt)
        ax = np.gradient(vx, self.dt)

        return self.t, x, vx, ax, p, v, a
