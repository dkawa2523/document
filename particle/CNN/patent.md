0. 管理情報（メタ）
	•	文書名: 半導体ウェハ付着粒子の空間×サイズ因子化解析に関する特許説明素材
	•	編集用Markdown: patent_material_semiconductor_ai.md￼
	•	文書種別: 発明開示書〜明細書たたき台素材
	•	想定発明名称案
	1.	粒子座標および粒径に基づくウェハ付着パターンの空間×サイズ因子化解析方法
	2.	ウェハ付着粒子の原因パターン推定システムおよびプログラム
	3.	低粒子数条件下での原因切り分けを行うウェハ粒子解析装置
	•	技術分野: 半導体製造、インライン/オフライン検査、AI、異常解析、要因推定、装置設計フィードバック
	•	対象読者: 審査官、弁理士、製造技術者、装置設計者、AI実装者
	•	技術文章の読み取り範囲
	•	先行会話で整理された技術内容:
HSSFNet-v2（Hierarchical Spatial×Size Factorized Network）、
ParametricMarkedMixture（説明・分解器）、
dual-grid（x-y 格子 + r-θ 格子）、
size-aware 前処理、
不確かさに応じた製造アクション分岐、
設計フィードバック。
	•	補助根拠として確認したデータ概要: 現在の particles.csv / samples.csv の実測監査。
	•	確認済みデータ概要
	•	samples.csv: 15,000サンプル
	•	particles.csv: 1,640,601粒子
	•	15クラス、各1,000サンプル
	•	サンプル当たり粒子数: 最小20、最大200、平均約109.37
	•	size profile: ln_narrow, ln_mid, mix_tail
	•	size model: lognormal, mixture
	•	現CSVでは size_assignment_scope=sample、size_coupling_policy=independent
	•	根拠資料
	•	現CSV監査結果￼
	•	前回の更新版仕様メモ￼
	•	HSSFNet-v2 コード骨格￼
	•	MarkedMixture コード骨格￼
	•	dual-grid target builder￼
	•	匿名化方針: 装置名、顧客名、製造ライン名、機密閾値は装置A、工程B、製造ラインX等に匿名化して記載。
	•	注意
	•	本資料は法的助言ではない。特許出願の実務判断は弁理士・弁護士・知財担当者による確認が必要。
	•	新規性・進歩性は先行技術調査前の仮説整理であり、断定ではない。
	•	現CSVは単一パターン学習に適している一方、空間局所×サイズ要因の完全な混合教師は将来拡張が望まれる。そこは本資料中で「一実施形態」「拡張実施形態」「要確認」として区別する。

1. 1ページ要約（発明の要旨／新規性の核3点／期待効果の定量）

発明の要旨

本技術は、ウェハ上に付着した粒子群について、各粒子の座標情報（少なくとも x-y または r-θ）と粒径情報 (size_um) を入力とし、これを 空間マップ と 粒径分布 に因子分解して推定する AI 解析技術である。
単なる画像分類や単純な点群分類ではなく、(i) ウェハ平面上の空間配置、(ii) 粒径分布、(iii) 形状パラメータ、(iv) 不確かさ、を同時に扱う。
第1段では、dual-grid に変換した count tensor から HSSFNet-v2 が セル×サイズbin×パターン の推定強度、パターン存在確率、幾何パラメータ、サイズ事前分布を出力する。
第2段では、ParametricMarkedMixture が pattern family ごとの空間カーネルと粒径モデルを用いて sample ごとに説明分解し、粒子責務、原因パラメータ、不確かさを返す。
これにより、低粒子数条件や複数要因の混在条件においても、単なる「異常あり/なし」ではなく、どの領域にどの原因パターンがどの程度存在し、その粒径分布がどうなっているか を可視化し、製造アクションおよび装置設計へのフィードバックに接続できる。

新規性の核3点
	1.	空間×サイズの因子分解
	•	単純な 2D 粒子マップや 3D 等方ボクセルではなく、x-y と r-θ の dual-grid を併用し、さらにサイズ方向を global size prior + local residual に分けて推定する点。
	•	これにより、空間パターンと粒径分布の双方を保持しつつ、低粒子数でも過学習しにくい。
	2.	識別器と説明器の二段構え
	•	第1段の HSSFNet-v2 が高速に cause map を出し、第2段の MarkedMixture が sample 単位の原因分解と粒子責務を与える点。
	•	単に「AIが分類した」で終わらず、原因仮説の説明性と監査可能性を確保する。
	3.	不確かさを介した製造アクション接続
	•	パターン存在確率、セル強度、粒子責務に対し、不確かさを算出し、しきい値に応じて
(a) 自動アクション、
(b) 追加計測、
(c) 人手レビュー、
(d) 再学習キュー投入、
に分岐させる点。
	•	AI推論結果を製造実務に安全に接続できる。

期待効果の定量（推定・要確認）

以下は推定値または開発目標値であり、実測値ではない。出願前に社内実験で要確認。

項目	期待効果	根拠区分
原因候補の絞り込み時間	50〜90%短縮	推定
目視レビュー工数	30〜70%削減	推定
不要な装置介入・誤アクション	10〜30%低減	推定
追加計測の優先順位付け精度	20〜40%向上	推定
低粒子数ウェハでの説明可能性	従来比で大幅改善	定性的
装置設計改善の仮説生成速度	1開発サイクル以上短縮の可能性	推定

2. 従来技術（背景）と先行技術カテゴリ

背景

半導体製造におけるウェハ上粒子の解析は、歩留まり改善、原因切り分け、装置保全、工程窓設定、装置設計最適化に直結する。
しかし従来は、粒子マップを画像化してパターン分類する手法、全ウェハの単一ラベル分類、統計量ベースのルール判定、人手による形状判定に依存することが多かった。
また、粒径情報は別帳票または補助統計として扱われやすく、空間配置と粒径の相互関係が学習・推定に十分に活かされないことがあった。

先行技術カテゴリ

カテゴリ	典型的アプローチ	想定される弱点
画像分類型ウェハマップ解析	粒子分布をラスタ画像に変換してCNN分類	低粒子数で情報が薄い、粒径を潰しやすい、粒子責務が出しにくい
点群分類・点群セグメンテーション	粒子座標列を点群として直接学習	空間場としての可視化や製造アクション接続に追加設計が必要
GNN系解析	粒子間近傍グラフで局所構造を学習	大域パターンと説明性の両立が難しいことがある
統計量・ルールベース	半径ヒストグラム、角度カバレッジ、密度など	混合要因や複雑形状に弱い
異常検知型	既知/未知異常のスコアリング	「何の原因か」までは出しにくい
装置診断ルール連携	判定結果を装置対応に結びつける	AIの信頼度や誤判定時の安全分岐が弱い

先行との差分候補
	•	単一の画像分類ではなく、粒子テーブルから空間×サイズ tensor を構築する点
	•	x-y と r-θ を並列表現として使う点
	•	サイズ方向を空間軸と等価にせず、global prior と local residual に分ける点
	•	識別器のあとに パラメトリック説明器 を置く点
	•	不確かさを製造アクションに直結させる点
	•	蓄積した pattern parameter を 工程改善・装置設計フィードバック に再利用する点

3. 従来の課題（発生条件、現行対策の限界、評価指標、制約）

発生条件
	•	粒子数が少なく、単純な画像化で情報が欠落する場合
	•	似た空間パターンであっても粒径分布が異なる場合
	•	同一ウェハ内で複数原因が重畳する場合
	•	エッジ近傍、局所ホットスポット、線状スクラッチ、リング、放射線状など、幾何が異なる場合
	•	装置差、メンテ直後/経時変化、材料ロット差などで分布が揺れる場合

現行対策の限界
	1.	画像化の限界
	•	低粒子数ではスパースであり、単純ラスタでは量子化誤差が大きい。
	•	粒径を色や輝度に埋めても、空間とサイズの意味が混ざる。
	2.	粒子単位分類だけでは現場説明が弱い
	•	「粒子AはクラスX」と出ても、製造技術者は「ウェハのどの領域にどの原因がどれだけ存在するか」を見たい。
	3.	全ウェハ単一ラベル分類の限界
	•	混合要因があると、primary label のみでは二次原因が見えない。
	4.	サイズ情報の扱いが弱い
	•	粒径を単なる補助統計にすると、「同じ位置だがサイズ違いで要因が違う」ケースに弱い。
	5.	信頼度不足
	•	AI出力に不確かさがないと、誤判定に基づく無駄な装置停止・点検・条件変更が起こりうる。

評価指標
	•	sample-level pattern presence F1 / AUROC
	•	cell-level count NLL / KL divergence
	•	geometry parameter error
	•	particle responsibility consistency（説明器）
	•	uncertainty calibration（ECE、閾値別精度など）
	•	製造アクションの precision / recall（要確認）
	•	レビュー工数削減率（推定）

制約
	•	ウェハは円形であり、x-y 格子には無効領域がある
	•	θ は周期境界を持つ
	•	粒子数が 20〜200 程度の sparse 条件がある
	•	現CSVは単一パターンが中心で、完全な mixed supervision は将来拡張が望ましい
	•	製造現場では誤判定コストが高く、不確かさ分岐が重要

4. 提案手法

4-1. システム構成（文章）

提案システムは、少なくとも以下の機能部を備える。
	1.	データ取得部
	•	検査装置または解析サーバが、粒子ごとの位置情報 (x_mm, y_mm, r_norm, theta_rad など) と粒径情報 (size_um) を取得する。
	•	必要に応じて sample-level メタ情報（パターンID、size profile、パターンパラメータ、工程ID、装置ID）を取得する。
	2.	前処理・特徴量生成部
	•	粒子テーブルを sample ごとに集約し、x-y 格子および r-θ 格子に変換する。
	•	同時に、サイズbinヒストグラム、平均 log-size、分散、ウェハマスク、エッジ距離などを作成する。
	3.	主学習器（HSSFNet-v2）
	•	dual-grid を入力し、
(i) パターン存在確率、
(ii) セル×サイズbin×パターンの強度テンソル、
(iii) サイズ事前分布、
(iv) 幾何パラメータ、
を出力する。
	4.	説明器（ParametricMarkedMixture）
	•	主学習器の上位候補ラベルと幾何初期値を受け、sample 単位で pattern family ごとの空間カーネル・サイズカーネルを最適化し、粒子責務と原因パラメータを返す。
	5.	不確かさ判定部
	•	パターン存在確率のエントロピー、ensemble 分散、xy/rt 整合性、説明器の責務エントロピー等から信頼度を算出する。
	6.	製造アクション接続部
	•	高信頼の場合は装置Aの点検候補、工程Bの条件レビュー候補、再計測対象、設計フィードバック候補などを提示する。
	•	低信頼の場合は追加計測、人手レビュー、再学習キューを指示する。
	7.	履歴・学習更新部
	•	推論結果、説明パラメータ、不確かさ、実際の処置結果を保存し、ドリフト監視および再学習に用いる。

参考コード構成

モジュール	役割	主I/O
src/synthlab/ml/data/csv_dataset.py	CSV読込と sample grouping	sample_row, particles
src/synthlab/ml/data/target_builder.py	dual-grid 特徴量/教師生成	x_xy, x_rt, y_xy, y_rt, presence, geom
src/synthlab/ml/models/hssfnet.py	HSSFNet-v2	mu_xy, mu_rt, presence_logits, geom_pred, size_prior_logits
src/synthlab/ml/models/marked_mixture.py	説明器	responsibility, params, nll
src/synthlab/ml/models/losses.py	Poisson/BCE/KL/L1 損失	各 head 用 loss
src/synthlab/ml/train/train_hssfnet.py	主学習器の学習	学習済み重み
src/synthlab/ml/train/fit_marked_mixture.py	sample 単位の説明分解	パターン説明結果

4-2. 入力データ仕様（種類/同期/欠損/特徴量/独自性候補）

入力データの種類

粒子単位データ（必須）
	•	sample_id
	•	particle_id
	•	x_mm, y_mm または r_norm, theta_rad
	•	size_um
	•	任意で source, tool_id, lot_id, timestamp

sample単位データ（推奨）
	•	label または pattern_id
	•	canonical_pattern_id
	•	n_particles
	•	pattern_params
	•	size_model
	•	size_params_json
	•	size_profile_id
	•	resolved_size_params
	•	seed_offset（synthetic 生成時）

将来拡張の混合教師（要確認）
	•	component_label_fine
	•	component_id
	•	labels_fine
	•	label_fine_primary
	•	component_size_params_json
	•	size_component_id
	•	size_component_name
	•	size_source

上記拡張列は、直前の技術整理では有効候補として扱ったが、現CSVでは一部未確認。よって本資料では「拡張実施形態」とする。

同期・整合
	•	粒子行は sample_id で sample 行と結合される。
	•	n_particles は粒子行数と整合していることが望ましい。
	•	x-y と r-θ は相互変換可能であることが望ましい。
	•	サイズ情報は粒子単位で保持されることが望ましい。

欠損時の扱い
	•	x_mm, y_mm が無い場合は r_norm, theta_rad から再構成
	•	r_norm, theta_rad が無い場合は x_mm, y_mm から再計算
	•	size_um 欠損時は当該粒子を
	•	除外、または
	•	近傍/サンプル分布で補完、または
	•	特別 bin に割当て
のいずれかを行う（要確認）
	•	sample-level size profile 欠損時は unknown クラスまたはフラグを付与

特徴量

x-y 格子特徴
	•	count(h,w)
	•	mean_log_size(h,w)
	•	std_log_size(h,w)
	•	size_hist(h,w,b)
	•	wafer_mask(h,w)
	•	r_norm_map(h,w)
	•	edge_dist_map(h,w)

r-θ 格子特徴
	•	count(r,t)
	•	mean_log_size(r,t)
	•	std_log_size(r,t)
	•	size_hist(r,t,b)
	•	r_coord(r,t)
	•	theta_coord(r,t)

全体特徴
	•	sample 全体の size hist
	•	radial hist
	•	theta hist
	•	粒子数
	•	パターンパラメータ（教師・補助タスク用）

独自性候補
	•	画像を直接入力せず、粒子テーブルから dual-grid tensor を構築する点
	•	サイズを空間軸と等価に扱わず、size bin / size prior / local residual に分ける点
	•	空間表現を x-y と r-θ の両方で持つ点
	•	将来的に component 単位のサイズ出所まで扱える設計にしている点

4-3. 教師データ/ラベル定義

現実装に適した教師

sample-level presence
	•	y_presence[k] ∈ {0,1}
	•	単一パターンデータでは正解クラスのみ1
	•	mixed データでは存在する pattern family を multi-hot 化

cell×sizebin×pattern count 教師
	•	Y_xy[k,b,h,w]
	•	Y_rt[k,b,r,t]
	•	現単一パターンデータでは、全粒子が sample label に属するとみなして集計
	•	mixed データでは component_label_fine で集計

geometry 教師
	•	pattern_params から取り出したパラメータ
	•	例:
	•	radius_ratio, width_ratio
	•	angle_center_rad, angle_width_deg
	•	offset_ratio, segment_half_length_ratio
	•	sigma_ratio, sigma_x_ratio, sigma_y_ratio
	•	start_r_ratio, end_r_ratio, theta_max
	•	非該当パラメータは mask して学習

size profile 教師
	•	size_model
	•	size_profile_id
	•	resolved_size_params から導く global size prior
	•	将来拡張として component-level size prior

説明器側ラベル

説明器は教師あり分類器ではなく、半教師あり/弱教師ありのパラメータ推定器として動作する。
初期 active label は主学習器の上位候補を用い、粒子ごとの responsibility を推定する。

一実施形態と拡張実施形態

形態	正解粒度	主用途
一実施形態	sample-level 単一ラベル + count map	単一パターン学習、サイズ事前分布学習
拡張実施形態	component-level mixed teacher	混合要因分解、粒子責務学習、強説明性

4-4. AIモデル仕様

4-4-1. 主学習器 HSSFNet-v2

基本構造
	•	xy_backbone: x-y 格子を入力とする 2D U-Net 系
	•	rt_backbone: r-θ 格子を入力とする 2D U-Net 系
	•	global_pool + MLP: sample 全体の要約ベクトル
	•	presence_head: pattern presence
	•	size_prior_head: global size prior
	•	geom_head: 幾何パラメータ回帰
	•	xy_intensity, xy_size_residual
	•	rt_intensity, rt_size_residual

数学的表現
各パターン k、サイズbin b、セル c に対して、推定強度を
	•	λ_xy(c,k) : 空間強度
	•	q(k,b) : global size prior
	•	δ_xy(c,k,b) : local size residual

として、

μ_xy(c,k,b) = λ_xy(c,k) × softmax_b(log q(k,b) + δ_xy(c,k,b))

とする。
r-θ 側も同様に μ_rt(c,k,b) を計算する。

この構造により、サイズ方向は「第3の物理空間」ではなく、空間強度に条件付く粒径分布として扱われる。

損失関数
	•	L_xy_count: PoissonNLLLoss による μ_xy と Y_xy の損失
	•	L_rt_count: PoissonNLLLoss による μ_rt と Y_rt の損失
	•	L_presence: BCEWithLogitsLoss
	•	L_geom: mask 付き L1 / Huber
	•	L_size_prior: KL divergence または cross entropy
	•	L_consistency: x-y と r-θ の整合項
	•	L_calib: optional calibration loss

総損失の一例:

L = αL_xy_count + βL_rt_count + γL_presence + δL_geom + εL_size_prior + ζL_consistency

不確かさ
少なくとも以下のいずれかを用いる。
	•	ensemble 分散
	•	MC dropout
	•	presence entropy
	•	x-y / r-θ 予測差
	•	説明器責務のエントロピー

出力
	•	pattern presence
	•	cause map（cell × pattern または cell × sizebin × pattern）
	•	geometry parameters
	•	global size prior
	•	uncertainty score
	•	optional particle posterior（後段復元）

4-4-2. 説明器 ParametricMarkedMixture

目的
	•	1枚のウェハに対して、
	•	どの pattern family が有効か
	•	その geometry が何か
	•	そのサイズ分布が何か
	•	各粒子がどの原因成分に帰属しやすいか
を求める。

モデル
各 active pattern k に対し、

log λ_k(x,s) = log A_k + log g_k(x; θ_k) + log h_k(s; φ_k)
	•	A_k: 成分強度
	•	g_k(x; θ_k): 空間カーネル
	•	h_k(s; φ_k): 粒径カーネル（例: lognormal）

空間カーネル例
	•	Ring
	•	Sector_Edge / Sector_Internal
	•	StraightLine
	•	CurvedScratch
	•	Spiral
	•	Hotspot_Iso / Hotspot_Elliptic
	•	Crescent_Edge / Crescent_Internal
	•	EdgeOrigin_InwardSpray
	•	EdgeSource_Bursty

これらはgenerator の厳密逆関数でなくてもよい。微分可能 surrogate kernel として実装できる。

推定
	•	HSSFNet-v2 の top-K 候補ラベルと geometry 初期値を受ける
	•	最尤推定または変分推定で A_k, θ_k, φ_k を更新
	•	粒子 n の責務 r_nk を算出
	•	quadrature によりウェハ全体積分を近似

効果
	•	cause map を説明可能なパラメータ表現へ落とせる
	•	低粒子数時でも強い構造事前により安定化しやすい
	•	装置設計・工程改善へ直結しやすい

4-4-3. 実装方法の具体化

推奨コード実装

src/synthlab/ml/
  data/
    csv_dataset.py
    target_builder.py
    pattern_specs.py
  models/
    hssfnet.py
    marked_mixture.py
    losses.py
  train/
    train_hssfnet.py
    fit_marked_mixture.py
  infer/
    infer_hssfnet.py
    explain_marked_mixture.py

推奨初期設定例（実装例）

設定名	値	備考
xy_h, xy_w	64, 64	x-y 格子
rt_r, rt_t	48, 64	r-θ 格子
size_bins	8	log-size bin
wafer_radius_mm	150.0	300mmウェハ相当の一例
size_min_um, size_max_um	0.05, 8.0	現データに合わせた一例
width	64	backbone width
global_dim	128	global MLP 次元
n_quad_xy	64	MarkedMixture の積分格子
max_steps	200	説明器の最適化反復数
lr	1e-2	説明器の初期学習率

上記は実装例であり、出願時は「所定の格子サイズ」「所定の分割数」と一般化可能。

主要クラス
	•	GridConfig
	•	JointGridTargetBuilder
	•	HSSFNetConfig
	•	HSSFNetV2
	•	MixtureFitConfig
	•	ParametricMarkedMixture

処理インタフェース例

item = dataset[i]
targets = target_builder.build(item)

out = model(
    x_xy=targets["x_xy"].unsqueeze(0),
    x_rt=targets["x_rt"].unsqueeze(0),
)

mixture = ParametricMarkedMixture(active_labels=top_labels)
loss = mixture.neg_log_likelihood(xy_ratio, size_um)
resp = mixture.responsibilities(xy_ratio, size_um)

製造アクション接続例
	•	presence > T1 かつ uncertainty < U1
→ 装置Aの点検候補提示、工程Bの条件再確認候補提示
	•	presence > T2 だが uncertainty ≥ U1
→ 追加計測を要求
	•	unknown_score > T3
→ 人手レビューまたは新規クラス候補として再学習キューへ投入

しきい値 T1,T2,T3,U1 は要確認。出願時は「所定閾値」として一般化するのが無難。

6. 提案手法による効果
	1.	低粒子数条件での安定性向上
	•	global size prior と local residual の分離により、セルごとの sparse histogram だけに依存しない。
	2.	原因マップの出力
	•	粒子単位の判定を超えて、ウェハ上のどこにどの原因があるかを提示できる。
	3.	粒径を含む要因切り分け
	•	同じ空間位置であっても、サイズ分布の違いを別要因として扱える余地がある。
	4.	説明性
	•	MarkedMixture により geometry と size parameter が明示される。
	5.	不確かさに基づく安全運用
	•	高信頼時のみ自動アクションとし、低信頼時は追加計測・レビューへ送ることで誤介入を抑制できる。
	6.	装置・工程・設計への接続
	•	pattern family の発生傾向を集約し、装置Aの保全、工程Bの条件見直し、部材形状の設計改善仮説へ接続できる。

7. 新規性・進歩性

新規性の観点

候補1

粒子テーブルから x-y と r-θ の dual-grid を生成し、cell×sizebin×pattern の count tensor を教師にする点。

候補2

サイズ方向を 3D 空間軸として等方的に畳み込まず、global size prior + local residual として因子分解する点。

候補3

識別器 HSSFNet-v2 と説明器 ParametricMarkedMixture を接続し、分類結果を geometry/size parameter と particle responsibility に落とし込む点。

候補4

不確かさに応じて製造アクション、追加計測、再学習を分岐させる点。

進歩性の観点
	•	画像分類、点群分類、GNN、統計ルールそれぞれ単独では、
「低粒子数」「サイズと空間の同時扱い」「説明性」「製造アクション接続」
を同時に満たしにくい。
	•	dual-grid + factorized size + explanatory fitter の組合せは、
目的に対して相補的であり、単なる設計事項の寄せ集めではない、という主張余地がある。
	•	さらに、不確かさを用いた分岐と履歴再学習を加えることで、単なる分類器から現場運用可能な解析基盤へ昇華している点を強調できる。

留意点
	•	真の新規性・進歩性は先行技術調査が必要であり、本節はあくまで差分候補。
	•	特に、
	•	ウェハマップ画像分類
	•	粒子点群分類
	•	装置診断AI
	•	uncertainty-aware decision
の各分野で要調査。

8. 実施例（最低2つ：代表ケース＋変形例。条件・手順・結果が書ける範囲で）

実施例1：代表ケース（現CSVによる単一パターン学習）

条件
	•	データ: samples.csv 15,000件、particles.csv 1,640,601粒子
	•	クラス数: 15
	•	粒子数: 20〜200 / sample、平均約109.37
	•	size profile: ln_narrow, ln_mid, mix_tail
	•	現状の size coupling: sample / independent

手順
	1.	sample_id ごとに粒子群を集約
	2.	x-y 格子 64×64、r-θ 格子 48×64、size bin 8 で tensor 化
	3.	pattern_params から geometry 教師を生成
	4.	HSSFNet-v2 を学習し、
	•	pattern presence
	•	cell×sizebin×pattern 強度
	•	geometry parameter
	•	global size prior
を出力
	5.	presence が高い sample について、MarkedMixture で geometry/size parameter を説明分解
	6.	uncertainty が高い sample は保留し、レビューまたは追加計測に回す

結果
	•	単一パターン条件下で、各 family の空間特徴と size profile を同時に学習可能である。
	•	pattern_params を補助教師に用いることで、単なるラベル分類よりも幾何に敏感な内部表現が得られる。
	•	mix_tail のような重尾サイズ分布に対し、size prior head を持つことで size-aware な予測が可能になる。

数値精度は要確認。出願用には「このような出力が得られる」実施例として記載し、別紙で評価結果を追補するのが望ましい。

実施例2：変形例（混合要因分解＋製造アクション）

条件
	•	compose 等により、1 sample 内に複数 pattern family が混在する mixed データを生成または収集
	•	粒子ごとに component_label_fine、必要に応じて size_source または component_size_params_json を付与
（仮定/要確認）

手順
	1.	mixed sample から Y_xy[k,b,h,w], Y_rt[k,b,r,t] を component 単位で作成
	2.	HSSFNet-v2 を mixed supervision で fine-tune
	3.	presence 上位 pattern を active set として MarkedMixture を初期化
	4.	particle responsibility と geometry / size parameter を最適化
	5.	uncertainty が低い場合は、
	•	装置Aの点検候補
	•	工程Bの条件レビュー候補
	•	設計変更候補
を出力
	6.	uncertainty が高い場合は、
	•	追加SEM計測
	•	追加装置ログ確認
	•	新規クラス候補キュー投入
を実施

結果
	•	1枚のウェハ内に複数原因が重なっていても、セル単位の cause map と 粒子責務 を同時に出力できる。
	•	これにより、単なる全体ラベルでは不明であった二次原因・局所原因を提示できる。
	•	製造アクションを pattern family ごとに分岐できる。

mixed データの詳細仕様は要確認。ただし、発明の思想としては十分に一貫している。

9. 図面（Mermaid／左→右）と図面説明（符号表も）

9-1. 図1：全体システム構成図

flowchart LR
  A10[10\n検査装置群] --> A20[20\n粒子テーブル取得]
  A20 --> A30[30\n前処理\ndual-grid生成]
  A30 --> A40[40\nHSSFNet-v2]
  A40 --> A50[50\n不確かさ判定]
  A40 --> A60[60\nMarkedMixture]
  A60 --> A70[70\n原因説明結果]
  A50 --> A80{80\n信頼度分岐}
  A80 -->|高| A90[90\n製造アクション出力]
  A80 -->|低| A91[91\n追加計測\nレビュー]
  A90 --> A95[95\n履歴DB]
  A91 --> A95
  A95 --> A96[96\n再学習\n設計FB]

図1の説明

検査装置群10から粒子テーブルを取得し、前処理30で dual-grid を生成する。
HSSFNet-v2 40 が cause map 等を推定し、不確かさ判定50および説明器60に入力する。
信頼度分岐80により、高信頼時は製造アクション90、低信頼時は追加計測/レビュー91へ送られ、結果は履歴DB95に蓄積され、再学習や設計フィードバック96に利用される。

9-2. 図2：処理フロー（S1〜Sn、不確かさ分岐、再学習まで）

flowchart LR
  S1[S1\n粒子データ取得] --> S2[S2\nsample単位集約]
  S2 --> S3[S3\nxy格子生成]
  S2 --> S4[S4\nrt格子生成]
  S3 --> S5[S5\n特徴量生成]
  S4 --> S5
  S5 --> S6[S6\nHSSF推論]
  S6 --> S7[S7\npresence\ncause map]
  S6 --> S8[S8\ngeom\nsize prior]
  S7 --> S9[S9\n不確かさ算出]
  S8 --> S10[S10\nMixture初期化]
  S10 --> S11[S11\n説明分解]
  S9 --> S12{S12\n高信頼か}
  S12 -->|Yes| S13[S13\n工程/装置\nアクション]
  S12 -->|No| S14[S14\n追加計測\n人手確認]
  S13 --> S15[S15\n結果記録]
  S14 --> S15
  S15 --> S16[S16\nドリフト監視]
  S16 --> S17[S17\n再学習/設計FB]

図2の説明

粒子データ取得後、sample 単位に集約し、x-y 格子および r-θ 格子を生成する。
HSSFNet-v2 により cause map と size prior 等を出力し、不確かさを算出する。
説明器は geometry/size parameter と粒子責務を推定する。
高信頼時は製造アクションへ、低信頼時は追加計測・人手確認へ送る。
結果は履歴化し、ドリフト監視と再学習、設計フィードバックに用いる。

9-3. 図3：AIモデル構造

flowchart LR
  B10[10\n粒子テーブル] --> B20[20\nTargetBuilder]
  B20 --> B21[21\nxy特徴]
  B20 --> B22[22\nrt特徴]
  B20 --> B23[23\nglobal特徴]
  B21 --> B30[30\nxy backbone]
  B22 --> B31[31\nrt backbone]
  B30 --> B40[40\nxy intensity]
  B30 --> B41[41\nxy size residual]
  B31 --> B42[42\nrt intensity]
  B31 --> B43[43\nrt size residual]
  B23 --> B44[44\nsize prior]
  B23 --> B45[45\npresence]
  B23 --> B46[46\ngeom head]
  B40 --> B50[50\nmu_xy]
  B41 --> B50
  B44 --> B50
  B42 --> B51[51\nmu_rt]
  B43 --> B51
  B44 --> B51
  B45 --> B60[60\nactive labels]
  B46 --> B61[61\ninit params]
  B60 --> B70[70\nMarkedMixture]
  B61 --> B70
  B70 --> B80[80\nparticle resp]
  B70 --> B81[81\nparam explain]

図3の説明

TargetBuilder20 が粒子テーブル10から xy特徴21、rt特徴22、global特徴23 を作成する。
xy backbone30 と rt backbone31 から空間強度およびサイズ残差を推定し、size prior44 と組み合わせて mu_xy50 および mu_rt51 を得る。
同時に presence45 と geom head46 が active label と初期パラメータを与え、MarkedMixture70 が粒子責務80と説明パラメータ81を出力する。

9-4. 図面キャプション案（特許向けの短文）
	•	図1: ウェハ付着粒子の空間×サイズ因子化解析システムの全体構成を示す図。
	•	図2: 粒子データ取得から不確かさ分岐、再学習および設計フィードバックまでの処理フローを示す図。
	•	図3: dual-grid を用いた主学習器と説明器の連携構成を示す図。

符号表

符号	名称
10	検査装置群 / 粒子テーブル入力
20	粒子テーブル取得 / TargetBuilder
30,31	backbone
40〜46	head群
50,51	出力強度テンソル
60	説明器入力情報
70	MarkedMixture
80,81	信頼度分岐 / 説明出力
90	製造アクション
95,96	履歴DB / 再学習・設計FB

10. 本技術により創出される新たな価値

10-1. 従来困難だった意思決定（何が初めて可能になるか）
	•	単一ラベル判定ではなく、ウェハ上の局所領域ごとの原因パターンの可視化が可能になる。
	•	粒径を含むため、「同じ位置だがサイズ分布が異なる」ケースを原因候補として分けて考えられる。
	•	AI結果をそのまま使うのではなく、不確かさを条件とした運用分岐が可能になる。
	•	geometry parameter を伴って結果を返すため、製造技術者や装置設計者が人間可読な形で解釈しやすい。

10-2. プロセス予測/開発価値
	•	時系列に蓄積すると、工程ドリフトや装置劣化の兆候を pattern family 単位で監視できる。
	•	size profile の変化を伴う場合、発生源の変化や工程条件の変化を早期に仮説化できる。
	•	pattern_params と推定結果の差を監視することで、synthetic-to-real のズレや計測系ズレの評価に使える。
	•	工程窓探索において、歩留まり低下の前駆信号として particle pattern の悪化を用いる余地がある（推定）。

10-4. 装置設計価値
	•	エッジ偏在、線状、放射状、局所ホットスポット等の発生傾向を装置単位で集約し、
ガス導入部、遮蔽板、搬送経路、エッジリング、排気構造等の設計改善仮説へ接続できる（例示）。
	•	単なる「欠陥数増加」ではなく、どの形の欠陥がどの粒径帯で増えているか を設計部門へ返せる。
	•	これにより、装置A/Bの差分比較や設計変更前後比較がしやすくなる。

10-5. 価値指標テーブル（領域×KPI×根拠）

領域	KPI	根拠
製造運用	レビュー工数削減率	cause map + uncertainty により優先順位付け可能
製造運用	誤アクション率	高信頼時のみ自動介入する構成
工程改善	原因仮説収束時間	geometry/size parameter を同時提示
工程改善	追加計測効率	低信頼 sample のみ選別できる
装置設計	パターン別発生比較	familyごとの説明結果を履歴化可能
装置設計	設計変更の効果検証	変更前後の pattern/size 変化を比較可能
AI運用	再学習必要性検知	ドリフト監視と uncertainty 履歴
品質保証	監査可能性	particle responsibility と parameter 記録

11. 請求項例（たたき台）

11-1. 独立請求項（方法）※発明の芯を1本にまとめる

【請求項1】
ウェハ上に付着した複数の粒子に関する粒子データを用いて原因パターンを推定する情報処理方法であって、
前記粒子データとして、少なくとも各粒子の位置情報および粒径情報を取得する工程と、
前記位置情報および粒径情報に基づいて、ウェハ平面に対応する第1格子表現と、極座標系に対応する第2格子表現とを生成する工程と、
前記第1格子表現および前記第2格子表現を機械学習モデルに入力し、パターン存在確率、空間セルごとのパターン強度、および粒径分布に関する推定値を算出する工程と、
前記推定値に基づいて、ウェハ上の領域ごとの原因パターン分布を出力する工程と、
を含み、
前記機械学習モデルは、空間強度と粒径分布とを因子分解して推定することを特徴とする情報処理方法。

11-2. 従属請求項（不確かさ分岐、装置差補正、ドリフト監視/再学習、工程窓推定、逆設計、設計FBなど）

【請求項2】
前記機械学習モデルが、前記第1格子表現に対する第1分岐と前記第2格子表現に対する第2分岐とを備える、請求項1に記載の情報処理方法。

【請求項3】
前記粒径分布に関する推定値が、パターンごとのグローバル粒径事前分布と、セルごとの局所残差分布とから構成される、請求項1または2に記載の情報処理方法。

【請求項4】
前記パターン存在確率、前記空間セルごとのパターン強度、および前記粒径分布に関する推定値に基づいて不確かさを算出し、前記不確かさが所定条件を満たす場合に、追加計測、レビュー、または再学習処理に分岐させる、請求項1〜3のいずれか一項に記載の情報処理方法。

【請求項5】
前記原因パターン分布の出力後に、パターン family に応じたパラメトリック説明モデルにより、形状パラメータ、粒径パラメータ、および各粒子の責務を推定する、請求項1〜4のいずれか一項に記載の情報処理方法。

【請求項6】
装置識別情報、工程識別情報、またはロット識別情報に基づいて装置差補正を行う、請求項1〜5のいずれか一項に記載の情報処理方法。

これは任意実施形態。現時点では要確認。

【請求項7】
推定結果の履歴に基づいてドリフトを監視し、所定条件成立時に前記機械学習モデルを更新する、請求項1〜6のいずれか一項に記載の情報処理方法。

【請求項8】
前記原因パターン分布または前記形状パラメータに基づいて、工程条件の調整候補または装置設計変更候補を出力する、請求項1〜7のいずれか一項に記載の情報処理方法。

【請求項9】
前記位置情報が直交座標および極座標の少なくとも一方を含み、他方を演算により求める、請求項1〜8のいずれか一項に記載の情報処理方法。

【請求項10】
前記粒子データが synthetic データを含み、前記 synthetic データに含まれるパターンパラメータを補助教師として用いる、請求項1〜9のいずれか一項に記載の情報処理方法。

11-3. 独立請求項（装置）

【請求項11】
ウェハ上に付着した粒子の原因パターンを推定する解析装置であって、
粒子データ取得部と、
第1格子表現および第2格子表現を生成する前処理部と、
空間強度および粒径分布を因子分解して推定する機械学習部と、
前記推定結果に基づいて原因パターン分布を出力する出力部と、
を備えることを特徴とする解析装置。

11-4. 独立請求項（システム）

【請求項12】
粒子検査装置と、請求項11に記載の解析装置と、推定結果履歴を蓄積する記憶装置と、を備え、
前記解析装置が、前記推定結果に基づいて追加計測指示、装置点検候補、工程条件調整候補、または設計フィードバック候補を出力することを特徴とするシステム。

11-5. 独立請求項（プログラム/記録媒体）

【請求項13】
コンピュータを、請求項1〜10のいずれか一項に記載の情報処理方法を実行するための各機能として動作させるプログラム。

【請求項14】
請求項13に記載のプログラムを記録したコンピュータ読取り可能な記録媒体。

11-6. クレーム設計メモ（広いクレーム/狭いクレーム、回避設計ポイント）

広いクレームの芯
	•	粒子の位置情報 + 粒径情報
	•	第1格子表現 + 第2格子表現
	•	空間強度 + 粒径分布の因子分解
	•	原因パターン分布の出力

狭いクレーム化しやすい具体要素
	•	x-y + r-θ
	•	global size prior + local residual
	•	HSSFNet-v2 + ParametricMarkedMixture の二段構え
	•	uncertainty gating
	•	synthetic pattern parameter を補助教師に使う点

回避設計されやすいポイント
	•	dual-grid を使わず single-grid だけにする
	•	size を因子分解せず、単に属性として連結する
	•	explanation model を省略する
	•	uncertainty を別システムで処理する

したがって
	•	独立請求項では抽象度を上げつつ、
「複数空間表現」「粒径分布の因子分解」「原因パターン分布出力」
を芯に置くのが望ましい。
	•	従属請求項で dual-grid、説明器、不確かさ分岐、synthetic 教師、設計FB を押さえる。

12. 追加情報（実務で重要：データ量、汎化、更新、監査、セキュリティ/ガバナンス）

データ量
	•	現在確認できる学習用基礎データは 15,000 sample、約164万粒子。
	•	単一パターン pretrain には十分な規模感だが、mixed supervision は別途必要。

汎化
	•	装置差、工程差、時系列ドリフトへの汎化は要検証。
	•	tool_id, process_id, recipe_id 等を domain embedding として入れる実施形態が有効な可能性がある（要確認）。

更新
	•	uncertainty 高値 sample、レビュー不一致 sample、設計変更後 sample を再学習キューに自動投入する構成が望ましい。

監査
	•	推論結果だけでなく、
input hash, model version, uncertainty, top-k cause, geom params, action suggestion
を保存することが望ましい。

セキュリティ/ガバナンス
	•	装置名、顧客情報、工程名は匿名化
	•	学習データ改変履歴、モデル更新履歴、レビュー結果のトレーサビリティ確保
	•	権限分離（装置操作権限と解析閲覧権限の分離）が望ましい

付録A. 用語集（略語/専門語）

用語	意味
dual-grid	x-y 格子と r-θ 格子を並列に用いる表現
HSSFNet-v2	Hierarchical Spatial×Size Factorized Network v2
MarkedMixture	空間位置と粒径マークを持つ混合説明モデル
cause map	ウェハ上の各領域における原因パターン分布
presence	ある pattern family が sample 内に存在する確率
size prior	pattern または sample に対する粒径分布の事前
local residual	グローバルな粒径事前からの局所補正
responsibility	各粒子が各原因成分に属する確率的重み
uncertainty gating	信頼度に応じて処理分岐する仕組み
synthetic teacher	合成データの生成パラメータ等を補助教師に使うこと

付録B. 追加で必要な情報（優先度：高/中/低。最大10項目）

優先度	項目	形式
高	mixed データで component_label_fine / component_id を付与できるか	Yes/No
高	size の原因帰属を sample 単位ではなく component 単位で持たせるか	選択式: sample / component / local
高	実運用で最終出力は cell map、particle label、両方 のどれを主に使うか	選択式
高	製造アクションの対象は何か	選択式: 装置点検 / 工程条件 / 追加計測 / 設計FB / 複数
高	不確かさ高値時の標準オペレーションを定義済みか	Yes/No
中	装置ID・工程ID・レシピID を学習に使えるか	Yes/No
中	ドリフト監視の周期	選択式: ロット毎 / 日次 / 週次 / 月次
中	追加計測手段の候補	選択式: SEM / 再検査 / ログ解析 / その他
低	design feedback の受け手部門	選択式: 製造 / プロセス / 装置設計 / 品質保証
低	出願上、強調したい価値の優先順位	選択式: 解析精度 / 説明性 / 製造自動化 / 設計FB