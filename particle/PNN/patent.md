0. 管理情報（メタ）

項目	内容
文書名	特許資料案（発明開示書〜明細書素材）
発明名称案	ウェハーパーティクル点群から潜在付着場および観測品質を推定する情報処理方法、学習方法、システム、及びプログラム
作成目的	先の検討内容を、新規性／進歩性／有用性が把握しやすい特許説明素材に整理すること
対象分野	半導体製造 × AI、ウェハーパーティクル解析、点群学習、合成データ活用
前提データ	添付 particles.csv、samples.csv、および前回具体化したコード骨格
添付データ確認結果	15,000サンプル、約165万粒子、15ラベル均等、1サンプル粒子数20〜200、size_model は lognormal と mixture
既存前提	現行は PointMLP 系のパターンラベル分類器（ユーザー提供情報）
本稿の位置づけ	技術説明素材および請求項たたき台。法的判断、公報網羅調査、権利化判断そのものは含まない
外部公開情報	PointNet、PointMLP、Set Transformer、Perceiver IO、Point-MAE、synthlab_scaffold 公開リポジトリを参照  ￼
注意	実測評価値、装置要因との対応、権利化余地は推定または要確認を明示

注記
本稿では、添付データ・作成コード・公開論文・公開リポジトリに基づく事項はその範囲で記載し、それを超える効果値・装置要因対応・特許性判断は [推定] / [仮定] / [要確認] を付して記載する。

⸻

1. 1ページ要約（発明の要旨／新規性の核3点／期待効果の定量）

発明の要旨

本発明案は、ウェハ上のパーティクル観測データを、単なるラベル分類対象ではなく順序のない粒子集合として扱い、当該粒子集合から
	1.	潜在的な clean 付着分布、
	2.	粒径の空間分布、
	3.	観測品質、
を同時推定する技術である。

具体的には、粒子の x/y 座標、r_norm、theta、粒径 size_um からなる点群を、順序不変の attention 系 set encoder に入力し、極座標グリッドの各セルを query とする構造化 decoder により、潜在密度場および潜在サイズ場を出力する。さらに、見落とし率、偽陽性率、位置ずれ量などの観測品質指標を同時に推定する。学習時には、clean な粒子点群から dropout・false positive・jitter を付与した複数の観測 view を生成し、同一 clean サンプルに対する複数観測の整合性を学習させる。Set Transformer は集合要素間相互作用を attention で扱う set encoder/decoder を備え、Perceiver IO は任意の structured output を query ベースで出力でき、Point-MAE は point cloud の masked pretraining を可能にするため、本件の構成要素として整合的である。  ￼

新規性の核3点
	1.	データ生成の芯
clean 粒子点群から、見落とし・偽点・位置ずれを人工付与して observed 点群を生成し、同時に latent clean field と観測品質ラベルを教師として持つ点。
→ 単なる data augmentation ではなく、観測劣化そのものを教師化している点が核。
	2.	入力表現の芯
入力を画像化せず、x/y 正規化・r_norm・sin/cos(theta)・log(size)・source flag を含む粒子集合特徴として扱う点。
→ 低粒子数・不規則配置・粒径差の情報を保ったまま学習できる。
	3.	出力設計の芯
出力をクラス確率だけにせず、潜在密度場 + 潜在サイズ場 + 総粒子数 + 観測品質 + 補助ラベルに分解した点。
→ 「何のパターンか」だけでなく、「どこに unseen mass がありそうか」「本当に sparse なのか、見落としなのか」まで扱える。

期待効果の定量

下表のうち、添付データと実装仕様から直接言える数値は事実ベース、改善幅は [仮定] とする。

項目	従来	本提案	コメント
主出力次元	15ラベル確率程度	403出力	192セル密度 + 192セルサイズ + 総数1 + 品質3 + 補助ラベル15
入力単位	画像または単純点特徴	粒子集合	低粒子数でも粒径・半径・角度を保持
学習 view 数	1サンプル=1観測が中心	1 clean = K観測	添付データ 15,000 clean に対し、理論上 15,000×K view
粒子数カバー範囲	現行制約下で識別可能例に偏り	20〜200粒子を包含	添付データ確認結果
観測品質出力	なし	3指標 + 総粒子数	miss / fp / jitter / total count
[仮定] 低粒子条件での実用性	ラベル誤判定または保留	field と品質で説明継続	クラス当て失敗時にも意思決定材料を保持


⸻

2. 従来技術（背景）と先行技術カテゴリ（従来技術はPointMLPによるラベル分類、後は先行研究）

現行の前提は、PointMLP によるパターンラベル分類であり、ウェハ付着マップまたは点群から pattern label を推定する構成である、というユーザー提供情報である。PointMLP 自体は point cloud を pure residual MLP で扱う軽量・高速な分類系として提案されている。  ￼

公開文献ベースでみると、先行研究カテゴリは概ね以下に整理できる。

カテゴリ	代表例	要点	本件との差
点群直接入力の基本系	PointNet	点群を直接入力し、順序不変性を尊重	主用途は分類・分割。観測品質の同時推定までは主眼でない
点群分類の軽量系	PointMLP	複雑な局所幾何抽出を使わず分類性能を追求	主に label 予測。latent field 出力ではない
集合相互作用の attention 系	Set Transformer	set encoder/decoder による要素間相互作用の学習	本件ではこれをウェハ点群 + field 推定に適用
任意構造出力の decoder 系	Perceiver IO	arbitrary structured outputs を query ベースで出力	本件では極座標セル query に具体化
point cloud 自己教師あり	Point-MAE	masked autoencoding により point cloud 表現学習	本件では synthetic 大量生成を生かす前学習として位置づけ
大規模点群の高性能系	PTv3	scale 重視、効率化と大規模受容野	本件では有力代替候補だが、今回は structured field 出力を優先

PointNet は点群を直接扱う初期基盤であり、PointMLP はその流れの中で軽量分類器として位置づけられる。一方、Set Transformer は集合入力の相互作用表現に強く、Perceiver IO は任意構造の出力生成に適しているため、本件の「粒子集合 → 潜在付着場」という設定により親和的である。Point-MAE は大量 synthetic を教師なし事前学習に転用しやすい。  ￼

また、synthlab_scaffold は wafer_particles を極座標＋粒径＋ラベルの点群として定義し、train / eval / predict、pattern_coverage、calibrate_search、labeling_apply 等を process として整理している。したがって、本件は単独モデルというより、既存 synthetic 基盤の上に構築される下流学習・推論の新手法としても説明できる。  ￼

⸻

3. 従来の課題（発生条件、現行対策の限界、評価指標、制約）

観点	従来の課題
発生条件	付着点数が少ない、観測ノイズがある、パターンが弱い、混合要因がある
現行対策	高信頼サンプルに限定したラベル分類、人手レビュー、閾値判定
限界	クラス確率しか出ず、「見えていないだけ」か「本当に sparse」かを区別しづらい
評価の偏り	Top-1 accuracy、F1 等のラベル指標に偏り、潜在分布の再現性や観測品質を直接評価しない
制約	実測では latent clean 真値が得にくい、誤検出・見落としの真値も得にくい
結果	低粒子条件で識別可能な pattern に運用が限定される

本件の技術課題は、低粒子・高ノイズ時でも意思決定に使える出力を返すことである。したがって、課題設定そのものを「pattern label を当てる」から、「潜在付着場と観測品質を推定する」に置き換えるのが中心思想である。

新たな評価指標としては、少なくとも以下が必要となる。

指標群	例
空間再現	latent density field の KL / Earth Mover 距離 / セル相関
サイズ再現	latent size field の MAE / quantile error
数量再現	total_count MAE / MAPE
品質推定	miss / fp / jitter の回帰誤差
下流判断	abstain の適合率、再計測推奨の hit rate
従来比較	auxiliary label の accuracy / macro-F1


⸻

4. 提案手法

4-1. システム構成（文章）

本提案は、以下の機能ブロックからなる。
	1.	clean データ準備部
添付 particles.csv と samples.csv を sample 単位に再編成し、各サンプルについて clean 粒子集合と clean teacher field を生成する。
	2.	観測劣化シミュレータ部
clean 粒子集合に対して dropout、size-dependent dropout、edge-biased dropout、false positive、positional jitter を付与し、observed 点群と品質ラベルを生成する。
	3.	粒子集合 encoder 部
observed 点群を set encoder に入力し、粒子間相互作用を反映した latent set 表現を生成する。推奨実装は Set Transformer 系とする。  ￼
	4.	構造化 field decoder 部
極座標セルごとの query を latent set に照会し、潜在密度場および潜在サイズ場を出力する。推奨実装は Perceiver IO 的 query decoderとする。  ￼
	5.	品質推定部
miss rate、false positive rate、jitter sigma、total count を推定する。
	6.	補助判定部
optional として pattern family または pattern label を補助出力する。ただし、主出力は latent field と品質である。synthlab_scaffold の labeling 文書では fine label は phenotype、cause hypotheses は ground-truth ではなく人手レビュー用 metadata と整理されているため、原因ラベルを主監督信号にしないのが安全である。  ￼

4-2. 特長量化仕様

本件の特徴量化の芯は、粒子を画像化せず、粒子単位 feature の集合として保持する点にある。wafer_particles の公開 schema も、r_norm、r_mm、theta_rad、size_um、label 等の粒子表を前提としている。  ￼

入力特徴量仕様

特徴量	定義	意図
x_norm	x_mm / wafer_radius_mm	ウェハ径差の吸収
y_norm	y_mm / wafer_radius_mm	同上
r_norm	粒子の正規化半径	edge / ring の表現
sin(theta)	角度の正弦	角度の周期性保持
cos(theta)	角度の余弦	角度の周期性保持
log1p(size_um)	粒径の対数変換	heavy-tail 安定化
size_source_flag	component 起源か sample 起源か	size 付与方法の違いを保持

出力 teacher 仕様

teacher	形式	例
latent density field	R x T	例: 8 x 24 = 192 セル
latent size field	R x T	セル内平均 log1p(size)
valid mask	R x T	size field の有効セル
total_count	scalar	clean 総粒子数
quality	3 scalars	miss / fp / jitter
optional label	class index	15ラベル

4-3. 学習データ生成仕様

添付データを起点とする clean teacher 生成
	•	particles.csv から sample ごとに粒子を group 化
	•	r_norm と theta_rad から極座標セルへ binning
	•	セル count を total count で正規化して density field を生成
	•	セル内平均粒径から size field を生成
	•	samples.csv の label、size_model、pattern_params、resolved_size_params をメタ情報として保持

観測劣化シミュレータ仕様

下記数値は今回コード骨格の実装例であり、設計例である。

劣化要素	実装例
base dropout	0.10〜0.70
fp rate	0.00〜0.20
jitter sigma	0.000〜0.020
size_drop_strength	0.0〜1.8
edge_drop_bonus	0.0〜0.25
false point 上限	64点

この設計により、1つの clean サンプルから複数の observed view を生成できる。よって、実測では得にくい「同一 latent 真値に対する観測条件差」を教師として利用できる。

split 設計

添付データに val/test split が明示されていないため、例実装では label + size_model + n_particles bin の層化で分割する。

split	件数
train	10,500
val	2,250
test	2,250

4-4. AIモデル仕様

推奨モデル

推奨モデル名（仮称）
SPF-Net: Synthetic-first Particle Field Network

採用理由
	•	Set Transformer は集合入力の順序不変性を保ちながら、attention で要素間相互作用を表現できる。  ￼
	•	Perceiver IO は structured output を query ベースで出力できるため、極座標セル単位の field 推定に適している。  ￼
	•	Point-MAE は point cloud の masked pretraining を通じて大規模 synthetic の活用余地を与える。  ￼
	•	PointMLP は高速・軽量な分類器として有力だが、主眼は classification であり、構造化 field 出力や観測品質推定を本質的に要求する今回の目的には中心構造としてはやや弱い。  ￼

実装例パラメータ

項目	例設定
input dim	7
hidden dim	128
attention heads	4
encoder layers	4
decoder layers	2
radial bins	8
angular bins	24
num cells	192
auxiliary labels	15

出力 head

head	出力	活用
density head	セル密度 logits	latent clean spatial field
size head	セル size value	latent clean size field
count head	scalar	total clean count
quality head	3 scalars	miss / fp / jitter
label head	logits	補助分類

学習損失

損失	役割
density KL	spatial field 学習
size masked L1	size field 学習
count loss	total count 学習
quality loss	miss / fp / jitter 学習
label CE	補助ラベル学習
consistency loss	同一 clean 由来の view 間整合

発明の芯として押さえるべき点

本件の芯は「AIを使うこと」自体ではなく、以下の組合せにある。
	1.	clean 粒子点群から観測劣化を人工付与して teacher を作ること
	2.	粒子集合を極座標・粒径を保持した set feature として扱うこと
	3.	latent field と観測品質を同時推定すること
	4.	同一 clean サンプルに由来する複数 observed view の整合を学習すること

⸻

6. 提案手法による効果

実測での改善幅は未取得のため、改善方向は [推定] とする。

効果	内容	種別
ラベル以外の説明可能性	latent density field により「どこに unseen mass がありそうか」を提示可能	事実ベース
観測品質の可視化	miss / fp / jitter を同時に出力可能	事実ベース
低粒子条件への強さ	ラベルが不安定でも field と quality は返せる	[推定]
synthetic 活用効率	1 clean から多数 observed view を作れる	事実ベース
再計測判断	低信頼観測を品質指標で切り分け可能	[推定]
下流保守・工程判断	edge 起源、line 起源、hotspot 起源などの空間傾向を保持	[推定]

本件は、従来の「分類器の精度向上」にとどまらず、分類不能でも価値が残る出力体系を持つ点に意味がある。これは製造実務上、保守・再計測・工程切り分けに直結する。

⸻

7. 新規性・進歩性

新規性の整理

[推定] 公開論文ベースで確認できるのは、PointNet/PointMLP による点群分類、Set Transformer による set interaction 学習、Perceiver IO による structured output、Point-MAE による自己教師あり pretraining である。これらは個別には知られているが、ウェハーパーティクル点群に対して
	•	clean 粒子集合から観測劣化付き observed 集合を生成し、
	•	極座標 latent density field と latent size field を teacher とし、
	•	観測品質を同時推定し、
	•	同一 clean 由来の複数観測の consistency で学習する、
という具体的組合せは、少なくとも本稿作成時に確認した公開論文・公開リポジトリ中には見当たらない。ただし特許公報調査は未実施で要確認である。  ￼

進歩性の整理

進歩性の論点は、単なるモデル置換ではなく、課題設定の変更にある。

進歩性候補1

従来は「pattern label を分類する」設計であるのに対し、本件は「latent clean field と観測品質を推定する」設計へ変える。
→ 出力設計が異なるため、解決する課題が異なる。

進歩性候補2

従来の augmentation は入力多様化に留まりがちだが、本件は観測劣化の種類と量そのものを教師ラベル化している。
→ miss/fp/jitter を同時学習する点が差異となる。

進歩性候補3

低粒子・高ノイズ時に exact particle を1点ずつ復元するのではなく、極座標 field に落とすことで安定な構造推定を狙う。
→ ウェハーパターンの ring / sector / edge / radial という幾何特性に合う。

進歩性候補4

wafer_particles の公開 schema と process 体系では、粒子表・samples 表・labeling_apply・pattern_coverage・calibrate_search・train/eval/predict が整理されている。本件はその上で、観測劣化 simulator と latent field 学習を新たに組み込む構成であり、repo の抽象設計に対する具体的技術実装として位置づけられる。  ￼

⸻

8. 実施例（半導体製造での事例と従来の課題、その効果を説明）

以下は 仮定実施例 であり、装置要因との対応は 要確認。synthlab_scaffold の labeling 文書でも cause hypotheses は ground-truth ではなく metadata 扱いである。  ￼

実施例1：装置Aの edge 起因粒子
	•	対象: 装置Aのウェハ検査結果
	•	観測: 粒子数が少なく、edge 近傍に散発的粒子のみ観測
	•	従来: PointMLP ラベル分類では C01_Uniform または C02_EdgeBiased が不安定
	•	本提案:
	•	density field が outer radial band に偏る
	•	quality head が高 miss_rate を示す
	•	total_count は observed より大きい
	•	効果: 「本当に少ない」のではなく「edge 側に latent mass があるが観測欠落が大きい」仮説を提示できる

実施例2：装置Bの scratch 系粒子
	•	対象: 搬送後のウェハ
	•	観測: 30点程度の sparse 点群
	•	従来: pattern label が line 系に確信を持てず、人手レビューへ回る
	•	本提案:
	•	density field が特定角度に細長く伸びる
	•	size field が特定帯域で偏る
	•	jitter が小さい一方、fp が低い
	•	効果: line/scratch family の疑いを補助ラベルに加え、保守担当が搬送経路を重点確認しやすい

実施例3：装置Cの hotspot 系粒子
	•	対象: 成膜または吐出系の局所異常
	•	観測: 中心外れの局所クラスターが低粒子数で観測
	•	従来: hotspot と uniform の境界が曖昧
	•	本提案:
	•	density field の局所ピーク
	•	size field の局所異常
	•	total_count と miss_rate の組み合わせ
	•	効果: 「局所起源」「再計測優先」「装置局所汚染点検」の判断を支援

⸻

9. 図面（Mermaid／左→右）と図面説明（符号表も）

9-1. 図1：全体システム構成図

flowchart LR
    A[101\nCSV入力] --> B[102\nClean前処理]
    B --> C[103\nClean cache]
    C --> D[104\n観測劣化生成]
    D --> E[105\n学習データ]
    E --> F[106\n集合Encoder]
    F --> G[107\nField Decoder]
    F --> H[108\nQuality Head]
    F --> I[109\nLabel Head]
    G --> J[110\n密度場出力]
    G --> K[111\nサイズ場出力]
    H --> L[112\n品質出力]
    I --> M[113\n補助ラベル]
    J --> N[114\n意思決定支援]
    K --> N
    L --> N
    M --> N

図1説明

図1は、本発明の一実施形態に係るウェハーパーティクル解析システムの全体構成例を示す。

9-2. 図2：処理フロー

flowchart LR
    A[201\n粒子取得] --> B[202\n特徴量化]
    B --> C[203\nClean field作成]
    C --> D[204\n劣化view生成]
    D --> E[205\n二視点入力]
    E --> F[206\n集合表現生成]
    F --> G[207\n密度場推定]
    F --> H[208\nサイズ場推定]
    F --> I[209\n品質推定]
    G --> J[210\n損失計算]
    H --> J
    I --> J
    E --> K[211\n整合損失]
    K --> J
    J --> L[212\n学習更新]
    F --> M[213\n推論]
    M --> N[214\n保守判断]

図2説明

図2は、本発明の一実施形態に係る学習および推論処理のフロー例を示す。

9-3. 図3：AIモデル構造

flowchart LR
    A[301\n観測点群] --> B[302\nPoint Embed]
    B --> C[303\nSet Encoder]
    C --> D[304\nLatent Set]
    D --> E[305\nGrid Query]
    E --> F[306\nCross Attend]
    D --> F
    F --> G[307\nDensity Head]
    F --> H[308\nSize Head]
    D --> I[309\nPool]
    I --> J[310\nCount Head]
    I --> K[311\nQuality Head]
    I --> L[312\nLabel Head]

図3説明

図3は、本発明の一実施形態に係る粒子集合入力から潜在場及び品質を生成する AI モデル構造例を示す。

9-4. 図面キャプション案（特許向けの短文）
	•	図1は、本発明の一実施形態に係るウェハーパーティクル解析システムの構成例を示す図である。
	•	図2は、本発明の一実施形態に係る学習及び推論処理のフロー例を示す図である。
	•	図3は、本発明の一実施形態に係る集合入力型 AI モデルの構造例を示す図である。

符号表

符号	名称	説明
101	CSV入力	particles.csv と samples.csv
102	Clean前処理	sample 単位の整理、teacher field 生成
103	Clean cache	clean 点群と教師場の保存
104	観測劣化生成	dropout / fp / jitter 付与
106	集合Encoder	順序不変 attention encoder
107	Field Decoder	極座標 query による構造化出力部
108	Quality Head	miss / fp / jitter 推定部
109	Label Head	補助ラベル出力部
110	密度場出力	latent spatial density field
111	サイズ場出力	latent size field
112	品質出力	観測品質指標
114	意思決定支援	再計測、保守、レビュー優先度判断


⸻

10. 本技術により創出される新たな価値

10-1. 従来困難だった意思決定（何が初めて可能になるか）

従来は、「分類できるか／できないか」が中心であり、分類不能時は人手レビューに戻りやすかった。
本技術では、分類不能であっても以下が可能になる。
	•	unseen mass の位置の推定
	•	observed が sparse なのか、miss が大きいのかの区別
	•	再計測の優先度付け
	•	保守点検箇所の絞り込み
	•	ラベルではなく field を根拠とした説明

10-2. 半導体製造装置でのウェハーパーティクル要因と久手での価値

「久手」は技術文脈上の意味が不明であり、『工程』の誤記の可能性が高いと推定して記載する。要確認。
また、要因との紐付けは 仮説であり、ground-truth ではない。  ￼

pattern 傾向	想定要因例	工程上の価値
edge biased / edge sector	edge 部材、シール、周辺流れ	edge 系異常の切り分け
straight line / curved scratch	搬送接触、擦れ、経路異常	搬送経路点検の優先化
radial lines	回転／放射方向の機械・流体要因	回転体・シャワー系確認
hotspot	局所汚染、局所漏れ、局所吐出	局所部位メンテ対象化
ring	対称な流れ・温度・保持系影響	周方向均一性起因の仮説生成

10-4. 復元、Clean化の価値

本件の clean 化は、「本当の粒子を1点ずつ再構成する」ことを必須としない。
むしろ、潜在付着場の推定を通じて、
	•	低粒子観測から空間傾向を復元する
	•	見落としと真の低発生を区別する
	•	size の空間偏りを見る
	•	quality を含めて結果を監査可能にする

点に価値がある。

10-5. 価値指標テーブル

指標	定義	従来	本提案	備考
Field 再現誤差	latent density field の KL 等	なし	あり	新規指標
Count 誤差	total_count MAE	なし	あり	新規指標
Quality 誤差	miss / fp / jitter MAE	なし	あり	新規指標
補助分類精度	macro-F1 等	あり	あり	従来比較可能
Abstain 品質	保留判断の適合率	低い	[推定] 高い	要評価
保守ヒット率	点検指示の妥当率	低い	[推定] 向上	要現場評価
説明性	空間分布で説明できるか	低い	高い	field 可視化可能


⸻

11. 請求項例（たたき台）

11-1. 独立請求項（方法）※発明の芯を1本にまとめる

【請求項1】
ウェハ上のパーティクルを表す複数の点データを取得し、
各点データについて、少なくとも位置情報、正規化半径情報、角度由来情報、および粒径情報を含む特徴ベクトルを生成し、
前記特徴ベクトルの集合を、順序不変性を有する集合入力型エンコーダに入力して潜在集合表現を生成し、
複数の空間セルに対応する query を用いて前記潜在集合表現を復号し、ウェハ空間における潜在粒子密度場および潜在粒径場を推定し、
さらに前記潜在集合表現に基づいて、少なくとも見落とし率、偽陽性率、および位置ずれ量を含む観測品質指標を推定し、
前記潜在粒子密度場、前記潜在粒径場、および前記観測品質指標を出力する、
ことを特徴とするウェハーパーティクル解析方法。

11-2. 従属請求項例1

【請求項2】
請求項1に記載の方法において、
前記角度由来情報は角度の正弦値および余弦値を含み、
前記粒径情報は粒径の対数変換値を含む、
ことを特徴とする方法。

11-3. 従属請求項例2

【請求項3】
請求項1または2に記載の方法において、
学習時に、clean な粒子点群から、見落とし、偽陽性、および位置ずれの少なくとも一つを付与した複数の観測点群を生成し、
同一の clean 粒子点群に由来する複数の観測点群に対して、前記潜在粒子密度場の整合性が高くなるようにモデルを学習する、
ことを特徴とする方法。

11-4. 独立請求項（システム）

【請求項4】
ウェハ上のパーティクル解析システムであって、
パーティクル点群データを取得する入力部と、
前記点群データから位置情報、正規化半径情報、角度由来情報、および粒径情報を含む特徴量を生成する前処理部と、
前記特徴量の集合を入力として潜在集合表現を生成する集合入力型エンコーダと、
複数の空間セルに対応する query に基づいて前記潜在集合表現から潜在粒子密度場および潜在粒径場を生成するデコーダと、
前記潜在集合表現から観測品質指標を生成する品質推定部と、
前記潜在粒子密度場、前記潜在粒径場、および前記観測品質指標を出力する出力部と、
を備えることを特徴とするウェハーパーティクル解析システム。

11-5. 独立請求項（プログラム/記録媒体）

【請求項5】
コンピュータに、
請求項1から3のいずれか1項に記載の方法を実行させるためのプログラム。

【請求項6】
請求項5に記載のプログラムを記録したコンピュータ読み取り可能な記録媒体。

11-6. クレーム設計メモ（広いクレーム/狭いクレーム、回避設計ポイント）

観点	設計メモ
広いクレーム	「順序不変 set encoder」「query による空間場出力」「品質同時推定」を軸にし、Set Transformer / Perceiver IO の固有名詞は請求項本文では避ける
狭いクレーム	極座標グリッド、sin/cos(theta)、log(size)、miss/fp/jitter、two-view consistency、synthetic corruption teacher を限定要素として従属化
進歩性補強	画像分類ではなく、粒子集合から latent field と quality を同時推定する点を強調
回避設計ポイント	raster 化して CNN で field を出す設計、graph network に置換、quality を別モデルに分離、synthetic teacher を使わない設計
防御案	「極座標セルに対応する複数 query」「観測劣化を教師とする学習」「latent field と quality の同時出力」を複数従属請求項で押さえる
要確認	公報ベースで、synthetic corruption teacher + field output + quality joint estimation の先行例有無を調査する必要あり


⸻

12. 追加情報（実務で重要：データ量、汎化、更新、監査、セキュリティ/ガバナンス）

項目	実務上の論点	現時点の整理
データ量	15,000 clean sample は supervised 初期学習に十分な規模	事実ベース
汎化	装置差、工程差、検査条件差への一般化が鍵	要 real calibration
更新	定期的に simulator 分布と label taxonomy を見直す必要	pattern_coverage / calibrate_search が適合しやすい  ￼
監査	quality 出力と field 可視化により判定根拠の監査性が向上	[推定]
セキュリティ	顧客名、装置型番、lot ID は匿名化して学習	本稿では匿名化前提
ガバナンス	cause hypotheses を教師にしない、または厳格に検証後に限定利用	labeling 文書と整合  ￼
モデル更新	synthetic-only 更新と real-calibrated 更新を分けて監査	要運用設計
失敗時挙動	abstain、再計測推奨、保守レビューへ送る導線が必要	[推定] 有効


⸻

付録A. 用語集（略語/専門語）

用語	意味
latent clean field	観測ノイズを除いた潜在付着分布
density field	極座標セルごとの付着比率
size field	極座標セルごとの粒径統計
miss rate	本来ある粒子が観測されなかった割合
fp rate	本来ない粒子が観測された割合
jitter	観測位置ずれ量
set encoder	順序のない集合入力を扱うエンコーダ
query decoder	出力セルごとの query で構造化出力を生成するデコーダ
auxiliary label	主出力ではなく補助的に出すラベル
phenotype	fine label としての観測パターン
cause hypothesis	要因仮説。metadata であり ground-truth とは限らない


⸻

付録B. 追加で必要な情報（優先度：高/中/低。最大10項目）

優先度	確認事項	回答形式
高	出願対象は 推論方法 が中心か、学習方法 まで含めるか	選択式：推論のみ / 学習のみ / 両方
高	実測データを用いた calibration を請求項に入れたいか	Yes / No
高	出力は latent field を主とし、label は補助でよいか	Yes / No
高	ウェハ径は固定か	選択式：300mm固定 / 複数径対応 / 不明
高	実運用で最重要な品質指標はどれか	選択式：miss / fp / jitter / total_count / abstain
中	「久手」は「工程」の誤記か	Yes / No / 別意図あり
中	装置起因仮説を claims に含めるか、明細書実施例のみに留めるか	選択式：claimsにも入れる / 実施例のみ
中	Point-MAE 的 pretraining まで今回の出願範囲に含めるか	Yes / No
低	family label と fine label のどちらを保護対象として重視するか	選択式：family / fine / 両方
低	ソフト単体出願か、装置システムとの組合せ出願も狙うか	選択式：ソフト単体 / システム併記 / 両方


⸻

この案のまま、次段では 「明細書の実施形態」体裁に寄せた文章化 と 請求項の広狭2系統の書き分け に進めるのが自然です。