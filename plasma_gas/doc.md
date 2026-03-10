了解しました。
いまの前提なら、開発の中心は swarm ソルバそのものではなく、swarm 計算を surrogate 学習用の再利用可能なデータ工場に変える周辺機能です。LXCat は electron-neutral cross section と swarm parameter の蓄積基盤で、BOLSIG+ は純ガス・混合ガスに対して E/N を軸に transport/rate coefficients を計算できます。BOLSIG+ の documentation では複数の collision set を順に読み込んで混合気体を構成でき、rate coefficient は <Qv> として直接出力されます。さらに、cross section は Boltzmann/Monte Carlo に使うなら「complete set」であることが重要です。 ￼

したがって、全体設計は 電子側の swarm 由来係数 と 重粒子側の拡散・輸送 と 空間場 surrogate を分離しておくのがよいです。RF・HF・時間変動電場や励起状態分布まで扱うなら LoKI-B を検証系または補助ソルバとして併設できます。重粒子の拡散側は Cantera の mixture-averaged / multicomponent / binary diffusion を別レイヤで管理しやすいです。係数管理では flux と bulk を混同せず、一般の plasma modeling では flux を主系列として持ち、bulk は swarm 比較用に並行保持するのが安全です。 ￼

以下では、機能部ごとの役割と、そのまま mermaid で貼れるワークフローの形でまとめます。

⸻

1. 推奨する全体構成

機能は大きく 7 つに分けると整理しやすいです。
	1.	断面積レジストリ
	•	目的: cross section を「ファイル」ではなく「物理モデル単位」で管理する
	•	入力: 断面積ファイル、出典、version、species、注記
	•	出力: xs_set_id, manifest, quality status
	2.	Swarm campaign manager
	•	目的: 混合比・圧力・温度・E/N 条件を自動展開して batch 実行する
	•	入力: gas ratio 範囲、E/N grid、p、Tg、growth model
	•	出力: raw swarm outputs, logs, run metadata
	3.	Transport / rate table factory
	•	目的: raw 出力から surrogate 用の係数テーブルを正規化して作る
	•	入力: swarm raw results
	•	出力: mu_e, D_L, D_T, k_ion, k_att, k_exc, mean_energy, optional EEDF descriptors
	4.	Gas encoder / feature store
	•	目的: “ガス名” を直接入れず、係数曲線を latent に圧縮する
	•	入力: 係数曲線群
	•	出力: gas_latent, normalized feature tensors
	5.	MLP surrogate
	•	目的: 低次元条件から transport/rate coefficients を即時計算する
	•	入力: gas_latent + p + Tg + E/N
	•	出力: 係数値 or 係数補正値
	6.	U-Net surrogate
	•	目的: 空間場を高速再構成する
	•	入力: geometry mask, boundary mask, control params, gas latent / coefficient maps
	•	出力: ne, ni, phi, source map, power deposition など
	7.	Validation / active learning
	•	目的: 誤差の大きい gas ratio / condition を追加計算して再学習する
	•	入力: 誤差マップ、不確かさ、失敗例
	•	出力: next run candidates, updated dataset

⸻

2. 実装の考え方

A. 断面積レジストリ

いま断面積データセットを保有しているなら、次に必要なのは「どの gas mixture を、どの cross-section family で計算したか」を再現可能に固定することです。
ここでは最低でも次を持たせるとよいです。
	•	xs_set_id
	•	species list
	•	source / contributor
	•	dataset version
	•	complete / partial flag
	•	elastic / effective / excitation / ionization / attachment の有無
	•	適用メモ
	•	benchmark status
	•	hash 値

ポイントは、混合比を変えるたびに dataset を別物として乱立させないことです。
代わりに、**「base species の cross section セット」+「mixture condition」**で run を再構成できるようにします。

B. Swarm campaign manager

既存の自動実行コードをそのまま生かし、追加するのは campaign 生成器です。
ここでやることは、たとえば
	•	gas ratio simplex のサンプリング
	•	E/N grid の定義
	•	p, Tg, growth model の切替
	•	実行失敗時の retry / quarantine
	•	run manifest の保存

です。

ここで重要なのは、1 run = 1 物理条件 とし、必ず run_id を採番して raw と postprocess を紐づけることです。

C. Transport / rate table factory

surrogate 学習では raw ログを直接使うより、学習しやすい係数表へ落としてから使う方が安定します。
おすすめは 2 系統です。
	•	電子係数表
	•	mu_e*N
	•	D_L*N
	•	D_T*N
	•	k_ion
	•	k_att
	•	k_exc_j
	•	mean_energy
	•	optional: EEDF descriptor
	•	重粒子側輸送表
	•	mixture-averaged diffusion
	•	multicomponent diffusion
	•	binary diffusion
	•	必要なら ion mobility table

電子側は swarm solver 起点、重粒子側は別 transport layer 起点に分けると、あとで gas 拡張が楽です。

D. Gas encoder

ここが、ガス変更コストを下げる核心です。
gas 名を one-hot で入れるのではなく、係数曲線そのものを latent 化します。

実務的には次の流れがよく機能します。
	•	共通 E/N grid に再サンプリング
	•	log scaling / clipping
	•	各係数チャネルを連結
	•	PCA または 1D autoencoder
	•	16〜64 次元の gas_latent を作成

これで「Ar/N2 70/30」と「Ar/N2 60/40」は別カテゴリではなく、latent 空間上の近い点として扱えます。

E. MLP surrogate

MLP はまず transport surrogate に使うのが最も成功しやすいです。
つまり、いきなり空間場全部を学習する前に、
	•	入力: gas_latent + p + Tg + E/N
	•	出力: mu_e, D_L, D_T, k_ion, ...

を学習させます。

これにより、U-Net 側は「ガスごとの物性変化」を MLP から受け取るだけで済みます。

F. U-Net surrogate

U-Net は 場の再構成専用に切るのがよいです。
入力候補は次です。
	•	geometry mask
	•	electrode / boundary mask
	•	imposed voltage or source parameters
	•	pressure / gas temperature
	•	broadcast した gas_latent
	•	または 2D に展開した coefficient maps

出力候補は次です。
	•	electron density
	•	ion density
	•	plasma potential
	•	electric field norm
	•	source term maps
	•	power deposition map

実装上は、gas latent を直接入れる方法と、MLP が作った係数を spatial map にして入れる方法の 2 通りがあります。
後者の方が物理的意味づけはしやすいです。

G. Validation / active learning

最後に必要なのは、どこを追加計算すべきかを自動で決めるループです。
最初から全 gas ratio を埋めるより、
	•	surrogate 誤差が大きい領域
	•	swarm 係数の曲率が大きい領域
	•	外挿っぽい mixture 条件
	•	学習が不安定な field pattern

だけを追加計算した方が効率的です。

⸻

3. レイヤ分離の基本方針

この案件では、データセットを 2 段に分けると運用しやすいです。

Dataset-A: swarm / transport surrogate 用

目的は 係数を当てる MLP の学習です。
	•	1 サンプル = 1 条件点
	•	入力: gas ratio, p, Tg, E/N
	•	出力: transport / rate coefficients

Dataset-B: field surrogate 用

目的は 空間場を当てる U-Net の学習です。
	•	1 サンプル = 1 snapshot または 1 case
	•	入力: geometry, boundary, operating condition, gas latent / coefficient maps
	•	出力: density / potential / source fields

この 2 段にしておくと、ガス変更時にまず Dataset-A を更新し、その後 U-Net の条件入力だけ差し替える運用ができます。

⸻

4. Mermaid: 全体ワークフロー

flowchart TD
    A[Cross section registry] --> B[Campaign generator]
    B --> C[Swarm batch runner]
    C --> D[Raw output parser]
    D --> E[QA and convergence check]
    E --> F[Transport and rate table factory]
    F --> G[Feature store]
    G --> H[Gas encoder]
    H --> I[MLP transport surrogate]
    I --> J[Coefficient service]

    G --> K[Field dataset builder]
    J --> K
    K --> L[Conditional U-Net training]
    L --> M[Field surrogate inference]
    M --> N[Validation and uncertainty]
    N --> O[Active learning selector]
    O --> B


⸻

5. Mermaid: 断面積レジストリ機能

flowchart LR
    A1[Import cross section files] --> A2[Normalize metadata]
    A2 --> A3[Assign xs_set_id]
    A3 --> A4[Check completeness flag]
    A4 --> A5[Register source and version]
    A5 --> A6[Attach benchmark status]
    A6 --> A7[Approved registry]

    A4 -->|incomplete or unclear| A8[Review queue]
    A8 --> A2

この機能で持つべき項目
	•	xs_set_id
	•	species
	•	source
	•	version
	•	complete_flag
	•	process_channels
	•	notes
	•	benchmark_case
	•	status

⸻

6. Mermaid: Swarm campaign manager

flowchart TD
    B1[Define gas family] --> B2[Sample gas ratio simplex]
    B2 --> B3[Define p and Tg grid]
    B3 --> B4[Define E/N grid]
    B4 --> B5[Select growth model]
    B5 --> B6[Generate run manifest]
    B6 --> B7[Launch swarm jobs]
    B7 --> B8[Collect logs and raw outputs]
    B8 --> B9{Converged?}
    B9 -->|yes| B10[Store raw run]
    B9 -->|no| B11[Retry or quarantine]
    B11 --> B12[Manual review]

実務上のポイント
	•	gas ratio は等間隔より Sobol / Latin hypercube が向きます
	•	run_id と xs_set_id を必ず分離します
	•	失敗 run を dataset に混ぜないために quarantine を作ります

⸻

7. Mermaid: Transport / rate table factory

flowchart TD
    C1[Raw swarm outputs] --> C2[Parse transport values]
    C2 --> C3[Parse rate coefficients]
    C3 --> C4[Parse mean energy and diagnostics]
    C4 --> C5[Tag flux and bulk separately]
    C5 --> C6[Resample to common E/N grid]
    C6 --> C7[Apply smoothing and sanity checks]
    C7 --> C8[Create coefficient tables]
    C8 --> C9[Export table and manifest]

    H1[Heavy species transport inputs] --> H2[Mixture averaged or multicomponent diffusion]
    H2 --> C9

この機能の主な出力
	•	electron_transport_table.parquet
	•	electron_rate_table.parquet
	•	heavy_transport_table.parquet
	•	table_manifest.json

⸻

8. Mermaid: Gas encoder / feature store

flowchart TD
    D1[Coefficient curves] --> D2[Resample on common grid]
    D2 --> D3[Log scaling and normalization]
    D3 --> D4[Concatenate physics channels]
    D4 --> D5[PCA or 1D autoencoder]
    D5 --> D6[Gas latent vector]
    D6 --> D7[Feature store]

推奨チャネル
	•	mu_e*N
	•	D_L*N
	•	D_T*N
	•	k_ion
	•	k_att
	•	主要励起 rate
	•	mean_energy

⸻

9. Mermaid: MLP transport surrogate

flowchart TD
    E1[Transport dataset] --> E2[Train val test split]
    E2 --> E3[Input encoder]
    E3 --> E4[MLP training]
    E4 --> E5[Positivity and smoothness checks]
    E5 --> E6[Model evaluation]
    E6 --> E7{Pass threshold?}
    E7 -->|yes| E8[Freeze model]
    E7 -->|no| E9[Add samples or features]
    E9 --> E1

推奨入出力
	•	入力
	•	gas_latent
	•	p
	•	Tg
	•	E/N または mean_energy
	•	出力
	•	mu_e
	•	D_L
	•	D_T
	•	k_ion
	•	k_att
	•	k_exc_j

使い方
	•	まずは table の近似器
	•	次に baseline table に対する residual corrector
	•	最終的に U-Net への係数供給器

⸻

10. Mermaid: U-Net field surrogate

flowchart TD
    F1[Field solver snapshots] --> F2[Build input tensors]
    F2 --> F3[Geometry mask]
    F2 --> F4[Boundary mask]
    F2 --> F5[Operating conditions]
    F2 --> F6[Gas latent or coefficient maps]

    F3 --> F7[Channel concatenation]
    F4 --> F7
    F5 --> F7
    F6 --> F7

    F7 --> F8[U-Net]
    F8 --> F9[Predicted plasma fields]
    F9 --> F10[Physics and residual checks]
    F10 --> F11[Accepted field dataset]

入力のおすすめ
	•	geometry mask
	•	electrode mask
	•	boundary condition mask
	•	pressure
	•	source condition
	•	gas_latent を broadcast
	•	もしくは mu_e, D_L, D_T, k_ion ... の map

出力のおすすめ
	•	ne
	•	ni
	•	phi
	•	|E|
	•	S_ion
	•	P_abs

⸻

11. Mermaid: active learning ループ

flowchart TD
    G1[MLP and U-Net predictions] --> G2[Error and uncertainty analysis]
    G2 --> G3[Select next gas ratios]
    G3 --> G4[Select next E/N or operating conditions]
    G4 --> G5[Run new swarm or field simulations]
    G5 --> G6[Update datasets]
    G6 --> G7[Retrain models]
    G7 --> G1


⸻

12. 実装順のおすすめ

着手順はこの順が安全です。
	1.	断面積レジストリを作る
まず xs_set_id 管理を固める。
	2.	swarm campaign manager を作る
既存の自動実行コードを manifest 駆動にする。
	3.	transport / rate table factory を作る
raw を surrogate 用に正規化する。
	4.	MLP transport surrogate を先に作る
ここで gas 変更問題の大半を吸収する。
	5.	U-Net を conditional 化する
gas latent または coefficient map で条件付ける。
	6.	active learning を追加する
後から必要領域だけ埋める。

⸻

13. この設計の要点を一言でいうと

「ガスごとに別 dataset を作る」のではなく、
「cross section → swarm coefficients → gas latent → field surrogate」
という 4 段構造にする、これが本質です。

この形にすると、想定ガスが増えても更新対象は主に
	•	断面積登録
	•	swarm campaign
	•	coefficient table
	•	gas latent

で済み、U-Net 全体を毎回作り直す必要がなくなります。

次に必要であれば、これをそのまま 実装タスク一覧 に落として、
ディレクトリ構成, json/yaml の manifest 例, 学習データ schema, MLP/U-Net の具体的な入出力テンソル定義 まで書き下します。