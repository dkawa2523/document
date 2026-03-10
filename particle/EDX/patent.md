結論から言うと、実装の主力は 階層的 Spatial×Size Factorized Network、説明責任と要因切り分けの主力は Marked-Mixture / Point-Process に分けるのが一番きれいです。
この repo は particles に r_norm / r_mm / theta_rad / size_um を持ち、x_mm / y_mm、component_label_fine、component_id を持てます。compose は各粒子に component_label_fine と component_id を付け、sample 側には labels_fine と label_fine_primary を書きます。さらに既定の size model は lognormal で、既定設定では 0.05–5.0 μm の範囲が置かれています。したがって、空間×サイズ×パターン の教師テンソルも、連続空間の marked process も、そのまま作れます。sample.label は primary label なので、主教師に使うべきではありません。  ￼

加えて、repo の process catalog には compose が明示され、fit_params は「実測/疑似実測からパターン主要パラメータを推定する」用途として将来枠が確保されています。なので、学習器は別の ml/ パッケージとして実装し、原因推定器は fit_params に寄せるのが自然です。  ￼

1. 階層的 Spatial×Size Factorized Network の具体化

1-1. 教師データの定義

まず作るべき教師は、粒子表から直接集約した 2 種類のテンソルです。
	•	Y_xy[h, w, b, k]: x-y セル h,w、size bin b、pattern k の粒子数
	•	Y_rt[r, t, b, k]: r-θ セル r,t、size bin b、pattern k の粒子数

ここで k は component_label_fine、必要なら instance レベルでは component_id を追加で持ちます。compose が component_id を粒子ごとに書いているので、将来的に 同ラベル複数成分 の instance-level 学習にも伸ばせます。size bin は repo の既定 lognormal 設定に合わせて、まずは log-spaced 8 bins を 0.05–5.0 μm 上に切るのが始めやすいです。空間正規化は r_norm を使うか、x_mm / y_mm を wafer 半径で割って [-1,1] に載せます。x/y が無いデータでも schema 側に polar_to_cartesian_mm があるので再構成できます。  ￼

実装はこんな前処理にします。

def build_joint_targets(df_particles, label2id, wafer_radius_mm,
                        H=96, W=96, R=48, T=96, B=8,
                        size_min=0.05, size_max=5.0):
    log_edges = np.linspace(np.log(size_min), np.log(size_max), B + 1)
    K = len(label2id)

    X_xy_count = np.zeros((H, W), np.float32)
    X_xy_sumlog = np.zeros((H, W), np.float32)
    X_xy_sumsq = np.zeros((H, W), np.float32)
    X_xy_hist = np.zeros((H, W, B), np.float32)
    Y_xy = np.zeros((H, W, B, K), np.float32)

    X_rt_count = np.zeros((R, T), np.float32)
    X_rt_sumlog = np.zeros((R, T), np.float32)
    X_rt_sumsq = np.zeros((R, T), np.float32)
    X_rt_hist = np.zeros((R, T, B), np.float32)
    Y_rt = np.zeros((R, T, B, K), np.float32)

    for row in df_particles.itertuples():
        x = row.x_mm / wafer_radius_mm
        y = row.y_mm / wafer_radius_mm
        r = row.r_norm
        th = row.theta_rad

        h = int(np.clip((y * 0.5 + 0.5) * H, 0, H - 1))
        w = int(np.clip((x * 0.5 + 0.5) * W, 0, W - 1))
        rr = int(np.clip(r * R, 0, R - 1))
        tt = int(np.clip((th / (2 * np.pi)) * T, 0, T - 1))

        log_s = np.log(max(row.size_um, 1e-6))
        b = int(np.clip(np.digitize(log_s, log_edges) - 1, 0, B - 1))
        k = label2id[row.component_label_fine]

        X_xy_count[h, w] += 1
        X_xy_sumlog[h, w] += log_s
        X_xy_sumsq[h, w] += log_s * log_s
        X_xy_hist[h, w, b] += 1
        Y_xy[h, w, b, k] += 1

        X_rt_count[rr, tt] += 1
        X_rt_sumlog[rr, tt] += log_s
        X_rt_sumsq[rr, tt] += log_s * log_s
        X_rt_hist[rr, tt, b] += 1
        Y_rt[rr, tt, b, k] += 1

    # mean/std channels
    X_xy_mean = X_xy_sumlog / np.maximum(X_xy_count, 1)
    X_xy_var = X_xy_sumsq / np.maximum(X_xy_count, 1) - X_xy_mean**2
    X_rt_mean = X_rt_sumlog / np.maximum(X_rt_count, 1)
    X_rt_var = X_rt_sumsq / np.maximum(X_rt_count, 1) - X_rt_mean**2

    return {
        "xy": np.concatenate([
            X_xy_count[..., None],
            X_xy_mean[..., None],
            np.sqrt(np.maximum(X_xy_var, 0))[..., None],
            X_xy_hist
        ], axis=-1),
        "rt": np.concatenate([
            X_rt_count[..., None],
            X_rt_mean[..., None],
            np.sqrt(np.maximum(X_rt_var, 0))[..., None],
            X_rt_hist
        ], axis=-1),
        "y_xy": Y_xy,
        "y_rt": Y_rt,
    }

1-2. モデル本体

このモデルは dual-grid + global prior + local size residual で組むのがいちばん安定します。
理由は、repo の既定ラベルが ring_narrow / ring_wide / edge_sector_left / edge_sector_right / scratch_horizontal / scratch_vertical / radial_lines / hotspot_center / random_* で、ring/sector/radial は r-θ が効き、scratch/hotspot は x-y が効くからです。  ￼

私なら構成はこうします。
	1.	xy_branch: 2D U-Net
入力は count, mean_log_size, std_log_size, size_histogram, r_norm_of_cell, edge_distance, wafer_mask
	2.	rt_branch: 2D U-Net
入力は同様。θ 方向は circular padding、r 方向は通常 padding
	3.	global_branch: sample 全体の集約特徴
N_total, global radial hist, global theta hist, global size hist を MLP へ
	4.	presence_head: labels_fine の multi-label presence
π_k = sigmoid(head(g))
	5.	size_prior_head: pattern ごとの global size prior
q_k[b] = softmax(head(g))
	6.	local_intensity_head: 各セルの pattern intensity
λ_xy[c,k] = softplus(head([F_xy[c], F_rt→xy[c], g]))
	7.	local_size_residual_head: 各セルの local size shift
δ[c,k,b] = head([F_xy[c], F_rt→xy[c]])
	8.	最終出力
μ_xy[c,k,b] = λ_xy[c,k] * softmax_b(log q_k[b] + δ[c,k,b])

ここで重要なのは、サイズ分布を global prior q_k[b] と local residual δ に分けることです。
低粒子数ではセル単位の size histogram はすぐスカスカになるので、pattern 別の global size prior を先に持たせ、その上に局所補正だけ乗せる方が強いです。これは「joint tensor を直接丸ごと予測」より安定しやすいです。count tensor 側でも、Poisson / negative-binomial を用いた非負因子化は sparse count と overdispersion に相性が良く、Poisson を multinomial augmentation に落とすと推論がかなり扱いやすくなります。  ￼

モデルの最小骨格はこうです。

class HSSFNet(nn.Module):
    def __init__(self, c_xy, c_rt, k_patterns, b_bins, d_model=128):
        super().__init__()
        self.xy_enc = UNet2D(c_xy, d_model)
        self.rt_enc = UNet2D(c_rt, d_model)
        self.global_mlp = nn.Sequential(
            nn.Linear(128, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU()
        )
        self.presence_head = nn.Linear(256, k_patterns)
        self.size_prior_head = nn.Linear(256, k_patterns * b_bins)

        self.intensity_head = nn.Sequential(
            nn.Conv2d(d_model * 2 + 16, d_model, 1),
            nn.GELU(),
            nn.Conv2d(d_model, k_patterns, 1),
        )
        self.size_resid_head = nn.Sequential(
            nn.Conv2d(d_model * 2, d_model, 1),
            nn.GELU(),
            nn.Conv2d(d_model, k_patterns * b_bins, 1),
        )

    def forward(self, x_xy, x_rt, gvec, xy_to_rt_index):
        f_xy = self.xy_enc(x_xy)               # [B, D, H, W]
        f_rt = self.rt_enc(x_rt)               # [B, D, R, T]
        f_rt_xy = sample_rt_to_xy(f_rt, xy_to_rt_index)  # [B, D, H, W]

        g = self.global_mlp(gvec)              # [B, 256]
        pres_logit = self.presence_head(g)     # [B, K]
        log_q = self.size_prior_head(g).view(-1, K, B)  # [B, K, B]

        gmap = expand_global_to_map(g, f_xy.shape[-2:])[:, :16]  # [B,16,H,W]
        lam_logit = self.intensity_head(torch.cat([f_xy, f_rt_xy, gmap], dim=1))
        lam = F.softplus(lam_logit)            # [B, K, H, W]

        delta = self.size_resid_head(torch.cat([f_xy, f_rt_xy], dim=1))
        delta = delta.view(x_xy.shape[0], K, B, *x_xy.shape[-2:])   # [B,K,B,H,W]
        log_q_map = log_q[:, :, :, None, None]                      # [B,K,B,1,1]
        p_size = F.softmax(log_q_map + delta, dim=2)               # [B,K,B,H,W]

        mu = lam[:, :, None] * p_size                               # [B,K,B,H,W]
        return {"mu_xy": mu, "presence_logit": pres_logit, "log_q": log_q}

1-3. 損失関数

私は最初はこう置きます。

\mathcal L
=
\mathcal L_{\text{count-xy}}
+
0.5\,\mathcal L_{\text{count-rt}}
+
\lambda_{\text{pres}} \mathcal L_{\text{presence}}
+
\lambda_{\text{cons}} \mathcal L_{\text{xy/rt-consistency}}
+
\lambda_{\text{smooth}} \mathcal L_{\text{smooth}}
	•	count-xy: Y_xy と μ_xy の Poisson NLL
	•	count-rt: Y_rt と μ_rt の補助 Poisson NLL
	•	presence: labels_fine multi-hot に対する BCE / focal BCE
	•	consistency: xy と r-θ を再投影した marginals の KL / L1
	•	smooth: λ_xy と δ の TV or Laplacian penalty

synthetic だけなら Poisson で十分ですが、実測に寄せる段階では negative binomial に切り替えるのが安全です。Hu らは sparse count tensors に対して、Poisson だけでは overdispersion に弱いと明示しています。  ￼

1-4. 学習時の実装上の罠

一番大きい罠は augmentation です。
edge_sector_left/right と scratch_horizontal/vertical があるので、回転・反転はラベル置換表込みで実装しないと教師が壊れます。ring_* や hotspot_center は回転不変ですが、方向つきラベルは不変ではありません。なので最初は augmentation を控えめにし、使うなら「変換後ラベル remap」を明示します。  ￼

二つ目の罠は outside-wafer です。
x-y grid は円外セルが大量に出るので、必ず wafer mask channel を入れ、loss も mask します。r-θ 側は外周外セルの問題がない代わりに θ wrap-around があるので circular padding を使います。これは実装で効きます。

三つ目は sparse cells です。
ゼロセルが大半になるので、loss は全セルを均等に見るより positive-heavy にします。具体的には zero cells を subsample するか、occupancy BCE を別 head にして hard negatives だけ見ると安定します。

1-5. 推論で出すもの

このモデルは分類器ではなく、以下を返すようにします。
	•	P(pattern k present | wafer)
	•	E[count in cell c, size bin b, pattern k]
	•	E[size distribution | cell c, pattern k]
	•	uncertainty(c,k) または deep ensemble variance

粒子単位が欲しければ、推論後に粒子 n が属するセル c_n と size bin b_n を見て

r_{nk}
\propto
\mu[c_n, b_n, k]

で particle posterior を復元できます。
つまり 出力は cell 単位で持ち、必要なら particle 単位に戻せる 形にしておくのが実務上いちばん扱いやすいです。

1-6. リポジトリへの組み込み

この案は repo にはこう差し込みます。

src/synthlab/ml/
  datasets/
    joint_grid.py
    collate.py
  models/
    hssfnet.py
    heads.py
  losses/
    poisson_nb.py
    consistency.py
  train/
    trainer.py
    metrics.py

conf/ml/
  hssfnet.yaml
  dataset/joint_grid.yaml
  train/default.yaml

scripts/
  train_hssfnet.py
  infer_hssfnet.py

joint_grid.py は compose の artifact を読んで tensor 化するだけにし、manifest から particles/samples を引けるようにします。pattern parameter 推定は process catalog の意図に合わせて、後述の marked-mixture 側を fit_params に寄せるのがきれいです。  ￼

2. Marked-Mixture / Point-Process の具体化

こちらは 根本原因の説明器 です。
理論上の出発点は marked Poisson process で、Taddy & Kottas は NHPP の likelihood が integrated intensity と sampling density に因数分解できること、さらに joint intensity φ(x,y) を使うと joint location-mark process / conditional mark distribution / marginal point process を一体で推論できることを示しています。continuous-domain の Cox process では、観測点集合の likelihood は
p(D|\lambda)=\exp\left(-\int_T \lambda(x)\,dx\right)\prod_n \lambda(x_n)
となります。  ￼

この問題に合わせると、1 枚の wafer 上の粒子集合 \{(x_n, s_n)\}_{n=1}^N を、複数原因成分の重ね合わせとして

\log p(D)
=
\sum_{n=1}^N
\log \left[
\sum_{k=1}^K \lambda_k(x_n)\,h_k(s_n\mid x_n)
\right]
-
\sum_{k=1}^K \int_W \lambda_k(x)\,dx

で書くのが自然です。
ここで λ_k(x) が空間強度、h_k(s|x) が size の mark density、z_n は latent component assignment です。これは exactly “同じ場所だが size が違うと原因が違う” を表現できます。  ￼

2-1. まず作るべき実装: 離散セル有限混合

実装の一歩目は continuous ではなく、cell×size_bin の有限混合 Poisson です。
Aglietti らは multi-task Cox process を regular grid 上の counts として扱い、pattern/task ごとの log intensity を latent functions の線形結合として置いています。repo の今回の用途では、pattern を “task” と見なすのがちょうどよいです。  ￼

モデルはこうします。

N_{c,b}\sim \mathrm{Poisson}\left(\Delta A_c \sum_{k=1}^{K}\lambda_{c,k}q_{k,b}\right)

\log \lambda_{c,k}
=
\beta_k + \sum_{q=1}^{Q} W_{kq}F_q(c) + U_k(c)
	•	c: 空間セル
	•	b: size bin
	•	F_q(c): shared latent spatial factors
	•	W_{kq}: pattern-specific mixing weights
	•	U_k(c): pattern固有 residual map
	•	q_{k,b}: pattern k の global size prior

これは 空間分布 と サイズ分布 を分けつつ、pattern 間で spatial basis を共有できます。ring_narrow と ring_wide、edge_sector_left/right のような近い構造があるので、この shared-factor はかなり効きます。multi-task Cox の論文でも、task intensities を latent functions の線形結合で持つ形が中核です。  ￼

このモデルは EM で素直に回せます。
まず
\mu_{c,b,k}=\Delta A_c\,\lambda_{c,k}q_{k,b}
として、

E-step
r_{c,b,k}
=
\frac{\mu_{c,b,k}}{\sum_j \mu_{c,b,j}}

M-step
\hat q_{k,b}
=
\frac{\sum_c N_{c,b}r_{c,b,k}}{\sum_{c,b'}N_{c,b'}r_{c,b',k}}

空間マップ λ_{c,k} を自由 map にするなら
\hat\lambda_{c,k}
=
\frac{\sum_b N_{c,b}r_{c,b,k}}{\Delta A_c}
で closed form に近く更新できます。smoothness を入れたいなら、この update を初期値にして TV/Laplacian 正則化つきで数 step 最適化すれば十分です。
これが 最も早く動く marked-mixture 実装 です。

最小コードはこんな形です。

class FiniteMarkedPoissonMixture(nn.Module):
    def __init__(self, C, K, B, Q=8):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(K))
        self.W = nn.Parameter(torch.randn(K, Q) * 0.01)
        self.F = nn.Parameter(torch.randn(C, Q) * 0.01)   # shared latent maps
        self.U = nn.Parameter(torch.zeros(C, K))          # residual maps
        self.logit_q = nn.Parameter(torch.zeros(K, B))    # size prior

    def forward(self, area):
        log_lambda = self.beta[None, :] + self.F @ self.W.T + self.U   # [C,K]
        lambda_ck = F.softplus(log_lambda)
        q_kb = F.softmax(self.logit_q, dim=-1)                          # [K,B]
        mu = area[:, None, None] * lambda_ck[:, :, None] * q_kb[None]  # [C,K,B]
        return mu

def poisson_mixture_nll(counts_cb, mu_ckb):
    mu_cb = mu_ckb.sum(dim=1)   # sum over K
    return (mu_cb - counts_cb * torch.log(mu_cb + 1e-8)).sum()

2-2. Bayesian / continuous に上げるなら GP latent factors

もしセル近似を弱めて、連続座標 と 不確かさ をもっとちゃんと持ちたいなら、latent factors F_q を GP にします。
Lloyd らの VBPP は continuous Gaussian-process-modulated Poisson process を、discretization なし・観測イベント数に線形スケールで扱う variational 推論を与えています。GP 側の実装は GPyTorch の ApproximateGP + VariationalDistribution + VariationalStrategy がそのまま使えますし、SVI の外側は Pyro なら Trace_ELBO ベースで回せます。  ￼

実装戦略は 2 通りあります。

A. GPyTorch 主体
Q 個の latent GP を ApproximateGP で作り、cell center または observed particle coords で評価して W で混ぜる。
これは連続空間補間がきれいです。

B. Pyro 主体
latent maps を learnable field + Gaussian prior で置き、SVI で posterior を持つ。
こちらは実装が楽です。

連続 GP を使うなら、最初から full continuous marked likelihood に行くより、まず cell-count Poisson likelihood で warm-start し、その後 particle-level continuous likelihood に切り替えるのが現実的です。Lloyd らも discretization の弱点を指摘しつつ、連続モデルは計算上の工夫が必要なことを前提にしています。  ￼

2-3. 一番説明しやすい実装: parametric geometry mixture

現場説明で最も強いのは、pattern family ごとに空間 kernel を持つ parametric geometry mixture です。
repo 側は既定 pattern として ring / sector / scratch / radial_lines / hotspot / random 系を持ち、ring_narrow には radius_ratio: 0.7、width_ratio: 0.03 が置かれています。size 側は既定で lognormal です。なので、少なくとも初期 prior は repo config に寄せられます。  ￼

私なら kernel はこう置きます。
	•	ring
g_{\text{ring}}(x)=\exp\left(-\frac{(\|x\|-\rho)^2}{2\omega^2}\right)
	•	sector
g_{\text{sector}}(x)=
\sigma\!\left(\frac{\psi-|\angle(x)-\alpha|}{\tau_\theta}\right)
\cdot
\sigma\!\left(\frac{r-r_{\min}}{\tau_r}\right)
\cdot
\sigma\!\left(\frac{r_{\max}-r}{\tau_r}\right)
	•	scratch
g_{\text{scratch}}(x)=
\exp\left(-\frac{d_{\text{line}}(x;\phi,b)^2}{2\omega^2}\right)
\cdot
\sigma\!\left(\frac{L/2-|u_{\text{line}}(x)|}{\tau_L}\right)
	•	hotspot
g_{\text{hotspot}}(x)=
\exp\left(-\frac{\|x-c\|^2}{2\sigma^2}\right)
	•	radial lines
複数 line kernel の和

そして
\lambda_k(x)=A_k\,g_k(x;\theta_k)
と置きます。
mark 側はまず
\log s \mid z=k, x \sim \mathcal N(\mu_k + \delta_k(x), \sigma_k^2)
で十分です。低粒子数では δ_k(x) を消して global lognormal だけにした方が安定します。

ここで重要なのは、これは generator の exact inverse とは限らない ことです。
私は generator の sampling law をここでは完全には検証していないので、上の kernels は “微分可能な推定用 surrogate” です。もし generator と完全一致の inverse fitting が必要なら、generate.py の各 pattern sampler をそのまま尤度側に写すべきです。

2-4. EM 実装を particle-level で書くとこうなる

粒子列 (x_n, s_n) を直接使う最小版は次です。

class ParametricMarkedMixture(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.log_amp = nn.Parameter(torch.zeros(K))
        self.mu = nn.Parameter(torch.zeros(K))           # for log size
        self.log_sigma = nn.Parameter(torch.zeros(K))
        # plus pattern-specific theta_k parameters
        self.params = nn.ParameterDict({
            "ring_rho": nn.Parameter(torch.tensor([0.7])),
            "ring_logw": nn.Parameter(torch.tensor([-3.5])),
            # sector / scratch / hotspot params ...
        })

    def spatial_log_kernel(self, x, k):
        # return log g_k(x; theta_k)
        ...

    def mark_logprob(self, log_s, x, k):
        mu = self.mu[k]
        sigma = F.softplus(self.log_sigma[k]) + 1e-4
        return -0.5 * ((log_s - mu) / sigma) ** 2 - torch.log(sigma)

    def log_component(self, x, log_s):
        out = []
        for k in range(K):
            out.append(
                self.log_amp[k] + self.spatial_log_kernel(x, k) + self.mark_logprob(log_s, x, k)
            )
        return torch.stack(out, dim=-1)  # [N,K]

EM はこうです。

for it in range(max_iter):
    # E-step
    log_comp = model.log_component(x, log_s)         # [N,K]
    r_nk = torch.softmax(log_comp, dim=-1)

    # M-step (mark parameters closed-form-ish)
    w = r_nk.sum(dim=0) + 1e-8
    model.mu.data = (r_nk * log_s[:, None]).sum(dim=0) / w
    var = (r_nk * (log_s[:, None] - model.mu.data[None, :])**2).sum(dim=0) / w
    model.log_sigma.data = torch.log(torch.sqrt(var + 1e-6))

    # spatial parameters / amplitudes: gradient step on weighted NLL
    loss = marked_pp_nll(model, x, log_s, area_integrator)
    opt.zero_grad()
    loss.backward()
    opt.step()

marked_pp_nll は

-\sum_n \log \sum_k \lambda_k(x_n)h_k(s_n|x_n) + \sum_k \int_W \lambda_k(x)\,dx

です。積分は x-y grid 上の quadrature で十分です。
pattern priors を使うなら、ring_narrow は ρ ≈ 0.7R、ω ≈ 0.03R 付近に Normal prior を置き、size は lognormal prior を置きます。  ￼

2-5. Fully nonparametric joint mark-location 版

Taddy & Kottas の一番重要な示唆は、joint location-mark intensity を直接持つと、conditional mark distribution と marginal point process を一体で扱えることです。実装的には、cellized version なら

\phi_{c,b,k}
=
\Lambda_k \, f_k(c,b)

として joint density f_k(c,b) を mixture で持てます。
これは cell × size_bin × pattern の非負因子化そのものなので、HSSFNet の出力を warm-start に使えます。 q_k[b] と λ_{c,k} を分けて学習したあと、必要なら f_k(c,b) へまとめて fine-tune する流れです。Taddy & Kottas は joint location-mark mixture が mixed data-type marks を柔軟に扱えることも強調しています。  ￼

3. 実際のおすすめ構成

私なら、実装順はこうします。

第1段: HSSFNet を作る

これは 精度担当 です。
synthetic compose から教師テンソルを作り、cell × size_bin × pattern の count / presence を予測します。学習は安定しやすく、高速です。repo の既定パターン群にかなり素直に乗ります。  ￼

第2段: Parametric Marked Mixture を作る

これは 原因説明担当 です。
各 wafer に対し、「ring が何本、半径はどこ、幅はどれくらい、サイズ分布はどうか」を posterior で返します。実運用で欲しいのはこのタイプのレポートです。

第3段: HSSFNet を initializer に使う

ここが肝です。
HSSFNet の λ_xy[c,k] と q_k[b] を使って、marked-mixture の初期責務 r_{nk}、pattern presence、初期 μ_k,σ_k を入れます。これで EM/VI がかなり安定します。逆に、marked-mixture の posterior を使って HSSFNet に uncertainty-aware distillation を掛けることもできます。

4. 評価方法

精度だけだとこの問題は危ないので、評価は 3 層にします。
	1.	教師一致
component_label_fine に対する particle posterior accuracy
Y_xy / Y_rt に対する count NLL
labels_fine に対する presence F1
	2.	原因説明妥当性
推定された ρ, ω, α, ψ, c, σ などが synthetic truth に近いか
	3.	posterior predictive checks
marked spatial point process の review が挙げる
cross/dot-type summary characteristics、mark-weighted summaries、mark correlation functions
を、観測と posterior predictive で比較する

特に 3 は重要です。
単に class accuracy が高くても、「サイズと空間の結びつき」が壊れていれば原因切り分け器としては危険です。marked process の summary で posterior predictive check を回す方が、本問題には合っています。  ￼

⸻

要するに、HSSFNet は cell × pattern × size_bin を高速・安定に出す実務モデル、Marked-Mixture / Point-Process はその出力を原因仮説と不確かさに変換する説明モデル として作るのが最善です。
もし 1 本だけ選ぶなら HSSFNet、本当に要因切り分けまで持っていくなら 2 段構えにします。

次に進めるなら、compose artifact から Y_xy / Y_rt を作る joint_grid.py と、ParametricMarkedMixture の最小 EM 実装の2本を先に固定するのがいいです。