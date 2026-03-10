具体化すると、私はこの問題を 1 wafer = 1 graph の 多タスク GNN として組みます。主タスクは粒子ごとの component_label_fine 推定、補助タスクは粒子ペアの same-component 推定、さらにグラフ全体の labels_fine 推定です。この repo は train.py / predict.py で model_name を registry から引いて model plugin を呼ぶ形になっており、現状の domain model は baseline_centroid と noop だけなので、src/synthlab/domains/wafer_particles/models/ に新しい GNN plugin を追加するのが自然です。さらに compose は粒子ごとに component_id と component_label_fine、サンプルごとに labels_fine と components_json を出しているので、教師信号はかなり揃っています。  ￼

まず採るべき実装スタック

第一候補は PyTorch + PyTorch Geometric (PyG) です。PyG は Data / HeteroData / Batch を持ち、point-cloud 向けの DynamicEdgeConv、GravNetConv、PointTransformerConv、hetero 用の HeteroConv、global-local を混ぜる GPSConv、説明用の torch_geometric.explain を備えています。DGL でも実装可能ですが、今回のような「粒子点群 + 幾何特徴 + hetero anchor node」構成は PyG の方が部品が揃っていて速いです。PyG の current docs では point cloud processing tutorial も明示されています。  ￼

周辺は Lightning を学習ループ整理に、TorchMetrics を F1・confusion matrix・calibration に、scikit-learn を leakage を避けた group split に、Optuna をハイパラ探索に使うのが実務的です。Lightning は LightningModule / Trainer を提供し、TorchMetrics は calibration error を含む多くのメトリクスを持ち、scikit-learn には GroupKFold と StratifiedGroupKFold があります。Optuna は define-by-run で探索空間を柔軟に書けます。  ￼

グラフ定式化

Lite 版

最初は homogeneous graph で十分です。1 wafer を 1 個の Data にし、粒子を node、粒子間近傍を edge にします。教師は y_fine = component_label_fine、補助教師は comp_id = component_id、グラフラベルは y_graph = multi-hot(labels_fine) です。repo の compose 出力とそのまま対応します。  ￼

本命版

本命は heterogeneous graph です。node type を particle, radial_bin, angular_bin, global に分けます。particle は実粒子、radial_bin は半径帯、angular_bin は角度帯、global は wafer 全体コンテキストです。PyG の HeteroData と HeteroConv はこういう edge type ごとに別 GNN を当てる設計に向いています。  ￼

特徴量設計

この問題では、相対特徴だけでは不十分です。taxonomy に edge_sector_right/left/top/bottom と scratch_horizontal/vertical/diagonal があるので、絶対方位を消すと fine label が崩れます。したがって node feature は「絶対幾何」と「局所相対構造」の両方を入れます。  ￼

node feature

私なら最初の x_i は次です。

x_i = [
  x/R, y/R,
  r/R, sin(theta), cos(theta),
  1 - r/R,                      # edge distance
  log1p(size_um),
  z_size_within_wafer,
  size_rank_within_wafer,

  knn_min_dist/R,
  knn_mean_dist/R,
  count_in_radius(0.03R),
  count_in_radius(0.06R),
  count_in_radius(0.12R),

  local_r_std/R,
  local_theta_circular_var,

  lambda1, lambda2,             # local covariance eigvals
  linearity, isotropy,
  cos(2phi_local), sin(2phi_local),
  radial_align, tangential_align,
  line_residual_local
]

ここで効くのは次の解釈です。
	•	x/R, y/R, sinθ, cosθ は absolute direction を保持します。right/top/left/bottom と horizontal/vertical/diagonal の識別に必須です。
	•	1-r/R は edge-sector、hotspot_edge、random_edge_biased を分けやすくします。
	•	count_in_radius と knn_* は hotspot / cluster / random の密度差を拾います。
	•	local_r_std と local_theta_circular_var は ring vs sector vs random に効きます。
	•	lambda1, lambda2, linearity, isotropy, phi_local は scratch / radial_lines / hotspot の局所形状を拾います。
	•	cos(2φ), sin(2φ) にしておくと、線の向きの 180° 同値性を自然に扱えます。scratch はこの表現が効きます。
	•	radial_align = |u_local · r_hat|, tangential_align = |u_local · t_hat| は radial line と ring 的な局所方向の切り分けに効きます。

edge feature

edge は距離だけでは弱いです。NNConv など edge-conditioned な層を使う前提で、私は e_ij をこうします。

e_ij = [
  dx/R, dy/R, d/R,
  dr/R,
  sin(dtheta), cos(dtheta),
  r_bar * dtheta / R,           # arc-length like
  log(size_j / size_i),
  abs(size_j - size_i),

  unit_delta_dot_radial_i,
  unit_delta_dot_tangent_i,

  same_radial_bin,
  same_angular_bin,
  dist_center_to_line_ij / R
]

特に unit_delta_dot_radial_i と unit_delta_dot_tangent_i、dist_center_to_line_ij は効きます。radial_lines は中心を通る方向に伸びやすく、ring は接線方向に並びやすく、scratch は中心との関係が任意だからです。

global / anchor feature

repo の QC / taxonomy は r・theta の coverage と集中度を強く見ています。なので graph 全体の hist_r, hist_theta, n_particles, size_mean/std をそのまま global node に持たせるか、radial_bin / angular_bin anchor node に分散させるとかなり効きます。radial_bin = 8, angular_bin = 12 くらいから始めるのが無難です。taxonomy 側でも ring は広い角度 coverage、sector は狭い角度 coverage、hotspot は局所ピーク、random_edge_biased は edge 偏り、という期待則で書かれています。  ￼

グラフ構築

最初から 1 本の graph に固定しない方がいいです。私は multi-edge / multi-relation にします。
	1.	particle -> particle (knn_xy)
	•	k = 8 から開始。少粒子 wafer では k = min(8, N-1)。
	2.	particle -> particle (radius_xy)
	•	r = 0.06R と 0.12R の 2 スケール。
	3.	particle -> radial_bin
	•	自分の半径帯と隣接帯へ接続。
	4.	particle -> angular_bin
	•	自分の角度帯と隣接帯へ接続。
	5.	particle -> global
	•	全粒子を global node に接続。

この構成の利点は、局所幾何は particle-particle で拾い、大域的な「この wafer 全体は右半分が濃い」「edge 帯に偏る」「ring っぽく全周に広がる」を anchor 経由で自然に持ち込めることです。PyG の HeteroData / HeteroConv がちょうどこの用途向けです。  ￼

どの GNN 形が適切か

1) まず作るべき本命: Fixed multi-graph + NNConv + anchor branch

この問題で最初に実装するなら、私は NNConv 中心の geometry-aware GNN を推します。NNConv は continuous kernel / edge-conditioned convolution なので、上のような edge_attr をそのまま重みに反映できます。つまり「距離・角度・size 差・中心との幾何関係」を message passing に直結できます。これはこの問題にかなり合っています。  ￼

具体的には、
	•	local branch: NNConv -> PNAConv -> PNAConv
	•	context branch: HeteroConv で particle↔radial_bin, particle↔angular_bin, particle↔global
	•	fusion: local/context を concat
	•	heads: node family, node fine, edge affinity, graph multilabel

にします。

これがいい理由は、少粒子時に安定で、かつ 要因切り分けで説明しやすいからです。Dynamic KNN だけに頼るより、「どの edge 特徴が効いたか」を追いやすいです。

2) 精度ベースライン: ParticleNet / DynamicEdgeConv 系

accuracy ベースラインとしては ParticleNet 的な DynamicEdgeConv を必ず置きます。ParticleNet は unordered particle cloud 向けに DGCNN/EdgeConv を使う設計で、point cloud / particle cloud の表現に強いです。PyG にも DynamicEdgeConv があります。  ￼

ただし、少粒子・要因切り分けではこれ単独より、上の fixed graph branch と混ぜた方が安定します。なので私は dual-branch にします。
	•	Branch A: NNConv on fixed geometric edges
	•	Branch B: DynamicEdgeConv on current embedding space
	•	Fuse after 2 blocks

これで「明示幾何」と「学習された近傍」の両方を取れます。

3) sparse irregular geometry に強い候補: GravNet

GravNet は irregular detector geometry 向けに提案された層で、近傍を learnable latent space で作ります。sparse で局所密度がまばらなケースに相性が良く、PyG に GravNetConv があります。粒子数が少なく、しかも pattern が疎な wafer ではかなり有力です。  ￼

私の使い分けはこうです。
	•	点数が比較的多い: NNConv + DynamicEdgeConv
	•	点数がかなり少ない / 密度差が強い: GravNet
	•	大域依存が強い: GPSConv 追加

4) 大域コンテキスト強化: GPSConv / PointTransformerConv

複数パターン混合では、局所だけだと「edge hotspot なのか narrow sector なのか」が曖昧な粒子が出ます。こういうときは global attention が効きます。PyG の GPSConv は positional/structural encoding + local MPNN + global attention の 3 部構成ですし、PointTransformerConv は point cloud 向け attention 層です。私は 3 層目以降で 1〜2 ブロックだけ入れます。全層 attention にすると少粒子問題には少し大げさです。  ￼

5) 主 trunk にしない方がよいもの: EGNN / e3nn

EGNN や e3nn のような回転等変モデルは一般には魅力的ですが、この repo の fine label は absolute direction を含みます。edge_sector_right と edge_sector_top、scratch_horizontal と scratch_vertical を分けたいなら、主 trunk が強い回転不変・回転等変だと不利です。使うなら family-level 補助 branch に留めます。EGNN は回転・並進・反射等変、e3nn は O(3) 等変ライブラリです。  ￼

出力 head と損失

私は head を 4 本にします。
	1.	node family head
	•	ring / sector / line / hotspot / random / cluster / cox / ...
	2.	node fine head
	•	component_label_fine
	3.	edge affinity head
	•	その edge の 2 粒子が同じ component_id か
	4.	graph multilabel head
	•	wafer 全体の labels_fine

損失はこうです。
	•	L_family: CrossEntropyLoss(weight=class_weight, label_smoothing=0.05)
	•	L_fine: CrossEntropyLoss(weight=class_weight)
	•	L_edge: BCEWithLogitsLoss(pos_weight=...) か sigmoid_focal_loss
	•	L_graph: BCEWithLogitsLoss(pos_weight=...)

PyTorch の CrossEntropyLoss は class weight と label smoothing を持ち、BCEWithLogitsLoss は pos_weight を持ちます。focal は torchvision.ops.sigmoid_focal_loss が使えます。edge affinity は正例が少なくなりやすいので focal か pos_weight が有効です。  ￼

さらに実務では 階層整合制約 を入れます。fine label の親 family と family head の予測が矛盾しないよう、taxonomy から mask を作って fine logit を family ごとに制限します。repo の taxonomy は coarse / family / generator を持っているので、この制約が入れやすいです。  ￼

少粒子・要因切り分け向けの設計

この用途では、単に top-1 を返すだけでは足りません。不確かさ を返すべきです。私は推論出力を
	•	粒子ごとの family posterior
	•	粒子ごとの fine posterior
	•	edge affinity
	•	wafer 全体の multilabel posterior
	•	abstain / unknown flag

にします。

そして validation で temperature scaling か単純なしきい値調整を行い、ECE を見ます。TorchMetrics は calibration error を持っています。要因切り分けでは「断定しない勇気」が重要です。  ￼

また、少粒子時は component 推定を edge affinity → connected components で素朴に復元するだけでもかなり使えます。これで「この 4 粒子は同じ scratch 起因」「この 3 粒子は edge hotspot 起因」とまとまりが見えるからです。pairwise で苦しいなら、次段階で Object Condensation に切り替えるのが良いです。これは未知個数の object を sparse point/graph 上で復元するための考え方で、要因分解と相性が良いです。  ￼

データ分割で一番大事なこと

一番注意すべきは lineage leakage です。compose は component metadata に source_sample_id を保存しています。つまり train/test が違っても、同じ source sample 由来の component がまたがる可能性があります。少なくとも group-aware split が必要です。scikit-learn には GroupKFold と StratifiedGroupKFold があります。より厳密には、sample_id ↔ source_sample_id の二部グラフを作って connected component 単位で分けるのが安全です。  ￼

最初に作る 2 本

wafer_gnn_lite

最初の 1 本はこれです。
	•	graph: homogeneous
	•	edges: knn_xy + radius_xy
	•	node feat: 上の x_i
	•	edge feat: 上の e_ij
	•	trunk: NNConv -> PNAConv -> PNAConv
	•	heads: family, fine
	•	loss: L_family + L_fine

これは実装が軽く、少粒子でも比較的安定です。

wafer_gnn_mt_v1

次の本命はこれです。
	•	graph: HeteroData
	•	node types: particle, radial_bin, angular_bin, global
	•	local trunk: NNConv + DynamicEdgeConv
	•	context trunk: HeteroConv or GPSConv
	•	heads: family, fine, edge_affinity, graph_multilabel
	•	loss: L_family + L_fine + 0.5 L_edge + 0.3 L_graph

これが実務向けです。

初期ハイパーパラメータ

私は最初はこのくらいから始めます。
	•	hidden dim: 96
	•	message passing layers: 3
	•	dynamic branch layers: 2
	•	dropout: 0.1
	•	kNN: k=8
	•	radius: 0.06R, 0.12R
	•	radial bins: 8
	•	angular bins: 12
	•	optimizer: AdamW, lr 2e-3
	•	batch: 32 wafers
	•	early stopping monitor: node macro-F1 + graph multilabel F1 の加重和
	•	augmentation:
	•	粒子 dropout
	•	座標微小 jitter
	•	size 微小 jitter
	•	ただし global rotation は label remap を伴う場合だけ

最後の点は重要です。taxonomy が absolute direction を持つので、rotation augmentation をするなら right→top のように fine label も一緒に回す必要があります。  ￼

この repo に合わせた plugin 設定の形

train.py / predict.py は model config をそのまま plugin に渡すので、たとえばこういう設計で十分です。  ￼

wafer_particles:
  train:
    model:
      name: wafer_gnn_mt_v1
      graph:
        type: hetero
        knn_k: 8
        radius_scales: [0.06, 0.12]
        radial_bins: 8
        angular_bins: 12
      features:
        use_abs_xy: true
        use_polar: true
        use_size: true
        use_local_pca: true
        use_density: true
        use_anchor_nodes: true
      task:
        node_fine_target: component_label_fine
        node_family_target: taxonomy_family
        edge_target: same_component
        graph_target: labels_fine
      encoder:
        local_conv: nnconv
        dynamic_conv: dynamic_edgeconv
        global_conv: gps
        hidden_dim: 96
        num_layers: 3
      loss:
        edge_weight: 0.5
        graph_weight: 0.3
        focal_edge: true

ひとことで推奨をまとめると

この問題で一番良い出発点は、absolute orientation を残した node feature + 幾何 edge_attr + fixed graph branch + dynamic graph branch + component affinity head です。
つまり、「ParticleNet っぽさ」と「edge-conditioned geometry GNN」と「要因分解 head」を混ぜるのが最も実務的です。

次に進めるなら、wafer_gnn_lite と wafer_gnn_mt_v1 の PyTorch Geometric 雛形を、そのままこの repo の model plugin 形式に合わせて書き下ろします。