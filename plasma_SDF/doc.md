U-Net に限定してまとめると、今回いちばん筋がよく、しかも複雑になりすぎない構成は 「SDF-conditioned residual U-Net」 です。
核は 3 つだけです。
	1.	境界の位置は SDF で入れる
	2.	境界の値や物理は“空間マップ”にして入れる
	3.	U-Net 本体は標準寄りの強い形にとどめる

低圧プラズマでは、壁近傍の sheath は薄い非中性層で、RF sheath は典型的に 10 Debye 長以上でも装置サイズよりは十分小さく、局所的には表面法線方向の 1 次元問題として見やすいです。さらに低温プラズマ全体は reactor・sheath・feature・表面反応で長さスケールも時間スケールも大きく分かれます。U-Net はもともと多段の down/up-sampling と skip connection で「大域文脈」と「局所位置」を両立する構造で、近年の neural PDE solver でも multi-scale 表現と parameter conditioning の両立に使われています。今回の問題設定にはかなり相性がよいです。  ￼

提案する最終形

私なら、まずは次の 1 本で始めます。
	•	本体: 4 段の residual U-Net
	•	入力: 座標 + 役割別SDF + εr + 境界条件の拡張マップ
	•	条件注入: scalar 条件は encoder で埋め込み、各 residual block に scale-and-shift で入れる
	•	出力: φ, ne, ni, Te を直接出す
	•	拘束: 既知の電極電位だけ推論後に boundary overwrite
	•	loss: 全域の場誤差 + 壁法線電場 + 誘電体界面 jump + inlet/outlet balance

これで、MoE や attention、別 solver 連成を最初から持ち込まずに、境界近傍の難しさのかなりの部分を取れます。近年の “modern U-Net” 系では、ResNet block、GroupNorm、GELU、scale-and-shift conditioning が使われ、attention を足しても必ずしも改善しない例も報告されています。つまり、U-Net 本体はシンプルにして、境界表現を強くする方が今回の目的には合っています。  ￼

⸻

1. U-Net 本体はどう組むか

推奨 backbone
	•	4 resolution level の encoder-decoder
	•	各 level は 2 個程度の pre-activation residual block
	•	正規化は GroupNorm
	•	活性化は GELU
	•	kernel は基本 3×3
	•	skip connection は通常の U-Net のまま
	•	self-attention は初版では入れない

これは、U-Net の「粗いスケールで全体構造を掴み、細かいスケールで境界を戻す」という強みをそのまま活かしつつ、PDE surrogate でよく効く conditioning を載せた形です。original U-Net は context と localization を両立するために contracting path と symmetric expanding path を使い、modern U-Net 系はそこに GroupNorm や residual block、scale-and-shift conditioning を足しています。attention 追加が目立って効かなかった報告もあるので、まずは外すのが良いです。  ￼

座標チャネルは必須

普通の畳み込みは「どこにいるか」を本質的には持たないので、座標変換のような問題で失敗しやすいことが知られています。そこで U-Net 入力には最低でも x, y、軸対称なら r, z を明示チャネルとして入れます。これは CoordConv の考え方です。壁近傍サロゲートでは、sheath が「境界からの距離」と「向き」に依存するので、座標チャネルを入れるだけでも境界位置の再現はかなり安定します。  ￼

scalar 条件は入力末尾に足すだけでなく block ごとに注入

電圧、RF 位相、流量、組成、排気能力のような scalar 条件は、単に画素ごとにコピーした定数チャネルとして入れるより、embedding → scale-and-shift で各 residual block に入れる方が強いです。PDE 用の modern U-Net でも、Δt, Δx, ν のようなパラメータを embedding して up/down block に注入しています。FiLM はこの一般形で、conditioning 情報に応じて特徴量を feature-wise に affine 変換します。今回の境界条件可変問題にはこの形がそのまま使えます。  ￼

⸻

2. SDF を軸にした「汎用機能」

ここが今回の設計の中心です。
SDF は 「境界がどこにあるか」 を入れるための共通言語で、境界条件の種類が変わっても使い回せます。SDF の零レベル集合が境界を表し、各点は最短距離を連続値で持てるので、mask より情報量が多いです。SDF を使った geometry encoding は、形状が変わる PDE surrogate でも有効で、形状トポロジーが変わるケースにも拡張されています。  ￼

まず入れるべき役割別 SDF

最小構成なら次の 6 役割で十分です。
	•	d_ground
	•	d_power
	•	d_float
	•	d_dielectric
	•	d_inlet
	•	d_outlet

ここで大事なのは パーツ名ではなく役割で分けることです。
U-Net にとって重要なのは「この境界が接地金属なのか」「駆動電極なのか」「誘電体なのか」「流入口なのか」「排気口なのか」です。役割別に分ければ、問題ごとに境界数が変わっても channel 数を固定しやすくなります。  ￼

SDF から派生させる汎用機能

SDF から次の 3 つを作ります。
	•	法線
\mathbf n = \nabla d / \|\nabla d\|
	•	boundary band mask
壁から近い帯域だけを強調するマスク
	•	edge / corner distance
角部や電極端部が重要な場合のみ追加

低圧プラズマの sheath は局所的には表面法線方向の問題に近いので、法線は特に重要です。これを入れると、斜め壁や曲面壁でも「壁法線方向に似た sheath」を共有しやすくなります。  ￼

SDF の効果
	•	ない場合: U-Net は境界を mask の段差としてしか見られず、sheath の位置がにじみやすい
	•	ある場合: 境界までの距離と法線方向が明示され、壁近傍の sharp transition を保ちやすい

距離変換や SDF は、CNN の boundary-aware 学習で loss や auxiliary task に使うと改善が出やすいことも報告されています。つまり、SDF は入力にも loss 設計にも使える「共通の境界記述子」です。  ￼

⸻

3. 重要な考え方: SDF は「どこ」、境界物理マップは「何をするか」

ここが一番重要です。
SDF だけでは足りません。
SDF が教えるのは「どこに境界があるか」だけで、「その境界が何ボルトか」「誘電率はいくつか」「どのガスが何 sccm 入るか」「どれだけ排気されるか」は教えません。

逆に scalar 条件だけ入れても、「その値が空間のどこに効くか」は分かりません。

したがって U-Net には、
	•	SDF = どこにあるか
	•	boundary physics map = その境界が何をするか

の 両方 を入れるのが本質です。これは、geometry-aware な距離場と boundary-condition conditioning を組み合わせる考え方で、variable geometry・variable coefficient・variable boundary condition を扱う最近の PDE surrogate / operator learning の方向性とも整合しています。  ￼

⸻

4. 入力電圧の拡張

一番よい形

入力電圧は scalar の V をそのまま入れるのではなく、空間に拡張した電圧影響場として入れます。
具体的には、各ケースの電極電位と \epsilon_r を使って、charge-free な補助場

\nabla\cdot(\epsilon_r \nabla \Phi_{\mathrm{bc}})=0

を解き、その \Phi_{\mathrm{bc}} を 1 チャネル、さらに可能なら \nabla \Phi_{\mathrm{bc}} を 2 チャネル入れます。
U-Net はこの \Phi_{\mathrm{bc}} を見れば、「どの境界の電圧がどこへどう届くか」を空間場として読めます。低圧プラズマの RF sheath は表面近傍の非中性層で、その性質は sheath voltage に強く支配されます。既知境界を距離場で明示し、境界条件を正確に扱う方向は distance-function ベースの BC 埋め込みとも相性がよいです。  ￼

問題ごとに電極数が変わるとき

ここでも channel 数は増やしません。
各電極を別チャネルにするのではなく、全部まとめて 1 枚の \Phi_{\mathrm{bc}} に落とすのがポイントです。
すると「電極が 1 本でも 5 本でも入力 channel 数は同じ」です。U-Net にとってはこれがかなり重要です。

効果
	•	電圧 scalar だけ: 値は知っていても、どこに効くかは学習側が再構成する必要がある
	•	\Phi_{\mathrm{bc}} を入れる: 電極配置、形状、誘電率越しの結合、端部の強まりまで空間的に見える

特に RF 位相差や複数電極で効きます。  ￼

⸻

5. 誘電率・誘電体の拡張

必須入力

誘電体があるなら、少なくとも
	•	d_dielectric
	•	εr map
	•	dielectric interface normal

は入れるべきです。
理由は、これは単なる material classification ではなく 係数不連続を持つ interface problem だからです。界面問題では、係数、境界条件、jump 条件を明示的に扱う方がよく、operator learning 側でも coefficients や boundary conditions を input feature として与えると有効です。  ￼

U-Net 側の実装

入力には εr を 1 枚入れるだけでよいです。
複雑に one-hot を増やす必要はありません。SDF が界面位置を教え、εr が係数差を教えます。

loss では界面帯域に対して

[\phi]=0,\qquad [\epsilon \partial_n \phi]=0

を見ます。
もし dielectric charging の履歴依存が強いなら、拡張として surface charge \sigma_s を boundary state map として追加し、

[\epsilon \partial_n \phi]=\sigma_s

に変えます。誘電体材料や厚みが sheath electric field を変えること、誘電体表面帯電が表面電位やイオン入射条件を変えることは古くから知られています。  ￼

効果
	•	εr がない場合: 金属壁と誘電体壁が同じように見え、界面近傍の電場が平滑化される
	•	εr と界面 loss がある場合: 石英窓、絶縁スペーサ、誘電体被覆部の電場と sheath の立ち方が改善しやすい

⸻

6. ガス入力境界の拡張

何を入れるか

inlet には少なくとも
	•	d_inlet
	•	inlet 法線
	•	inlet type（mass-flow か pressure か）
	•	flow rate / \dot m
	•	gas composition
	•	gas temperature

を入れます。
低圧 ICP では、gas flow rate の増加で electron density が増え、electron temperature が下がる例が報告されています。つまり gas inlet は「中性粒子の境界条件」にとどまらず、最終的な plasma state も変えます。  ￼

どう U-Net に入れるか

おすすめは boundary value を空間マップへ拡張して入れることです。
たとえば、
	•	Q_in(x) = inlet 強度マップ
	•	Y_in^{(m)}(x) = 各主要種の組成マップ
	•	T_in(x) = inlet 温度マップ

のようにします。
これは inlet 境界上に値を置き、そこから内部へ滑らかに延ばしたマップです。U-Net には scalar ではなく、この spatial map を見せます。

rarefaction が強い場合

低圧では Knudsen 数が上がりやすく、Economou の tutorial でも Kn > 0.1 では kinetic model の方がより正確だとされています。したがって inlet 近傍に希薄流効果が強いなら、追加で neutral base field を 1–2 枚入れる価値があります。ただしこれは v2 でよく、初版は SDF + inflow maps で十分です。  ￼

効果
	•	pressure だけ: 「どこから何が入るか」が見えず、異なる inlet configuration を混同する
	•	inlet SDF + inflow maps: plume、非対称性、組成差、温度差を U-Net が空間的に区別できる

⸻

7. 排気境界の拡張

何を入れるか

outlet には
	•	d_outlet
	•	outlet 法線
	•	pump speed
	•	conductance
	•	throttle
	•	effective pumping speed S_{\mathrm eff}

を入れます。
真空工学では、pump の最大 pumping speed は aperture conductance を超えられず、throughput は conductance に制約されます。したがって排気は pressure out だけで表すより、吸い出し能力の空間分布として持たせた方が理にかなっています。  ￼

U-Net 側の実装

入力には 1 枚の suction map を入れます。
たとえば S_out(x) を outlet 境界から内部に延長したマップにして、さらに global scalar として S_{\mathrm eff} を condition encoder にも入れます。

効果
	•	outlet pressure だけ: pump や配管構成の差を失いやすい
	•	outlet SDF + suction map + S_{\mathrm eff}: どこがどれだけ吸えるかが見える

これは gas inlet と同じで、境界値を scalar のまま渡さず spatial map にするのが効きます。  ￼

⸻

8. 壁物性の拡張は optional だが、変わるなら入れる

SEE、反射率、sticking、recombination などがケースごとに変わるなら、それらも boundary property map として入れます。
理由は単純で、壁反射そのものが sheath を変えうるからです。実際、電子反射が強いと Debye sheath が弱まり、十分強い反射では消えることさえあると報告されています。つまり、壁物性が変わるのに geometry だけ入れても不十分です。  ￼

この拡張は必須ではありません。
ただし、「ある材料の壁だけ外れる」「同じ形状で表面処理が違うと外れる」という症状があるなら最優先で追加すべきです。  ￼

⸻

9. 出力は直接物理量でよい

今回の条件なら、出力は

\hat\phi,\;\hat n_e,\;\hat n_i,\;\hat T_e

を そのまま 出す方が分かりやすいです。
既知の Dirichlet 電位境界だけは推論後に boundary overwrite して厳密に満たします。正値制約が必要な n_e,n_i,T_e だけ softplus を使えば十分です。log 変換や差分出力にしなくても、入力側で境界を十分に見せ、loss 側で境界帯を見れば U-Net はかなり安定します。これは、distance-function による boundary handling と parameter-conditioned U-Net を組み合わせた実装上の推奨です。  ￼

⸻

10. loss は 4 本でよい

複雑にしすぎないなら、loss は次で十分です。

1) 全域の場 loss

L_{\mathrm{field}}
\phi, n_e, n_i, T_e の通常の誤差です。

2) 壁帯域の法線電場 loss

L_{\mathrm{wall}}
\sim \|\mathbf n\cdot\nabla \phi - \mathbf n\cdot\nabla \phi^\*\|^2

sheath では電場が局所的に表面法線方向を向きやすいので、壁近傍では値そのものより 法線方向勾配 を見る方が効きます。導関数情報を学習に使う Sobolev training は、値だけでなく導関数も使うことで精度と一般化を改善しうると報告されています。  ￼

3) 誘電体界面 jump loss

L_{\mathrm{int}}
\sim \|[\phi]\|^2 + \|[\epsilon\partial_n\phi]-\sigma_s\|^2

誘電体があるなら必須です。interface problem は単一の平滑な場として扱うと界面がにじみやすいので、jump を明示して学習させた方がよいです。  ￼

4) port balance loss

L_{\mathrm{port}}

総 inflow / outflow の整合です。
gas inlet と exhaust は pointwise 圧力より、port 全体の throughflow 整合の方が効きます。pump throughput が conductance に制約されることを考えると、これはかなり自然です。  ￼

optional

SDF/DTM を使った boundary-aware loss や auxiliary task は CNN を改善しやすいですが、実装依存性もあります。なので初版では「SDF 入力 + wall/interface/port loss」に留め、物足りなければ auxiliary head を足すのが安全です。  ￼

⸻

11. 学習は SDF バンドで行う

U-Net でも、学習データを全画素一様で投げると bulk に負けます。
そこで sample や patch は少なくとも
	•	wall band
	•	dielectric band
	•	inlet band
	•	outlet band
	•	bulk

に分けて層別化します。
sheath は薄いので、壁帯域を意識的に多く見せる必要があります。distance-map 系の学習は boundary-aware CNN の改善に効きやすく、U-Net のような multi-scale CNN でもその傾向があります。  ￼

⸻

12. これを「汎用機能」と「境界物理機能」に分けるとこうなる

汎用的な SDF 機能

これは 問題が変わっても常に使う部分です。
	•	x,y または r,z の座標チャネル
	•	役割別 multi-SDF
	•	SDF 由来の法線
	•	boundary band mask
	•	scalar 条件の embedding + scale-and-shift conditioning

この層の役割は、U-Net に geometry と location を理解させることです。CoordConv と SDF により「どこか」が入り、conditioning により「どんな条件か」が入ります。  ￼

境界物理の拡張機能

これは その問題にその境界物理がある時だけ足す部分です。
	•	電圧: \Phi_{\mathrm{bc}}, \nabla\Phi_{\mathrm{bc}}
	•	誘電体: \epsilon_r map と interface jump loss
	•	ガス入力: Q_in, Y_in, T_in の inflow maps
	•	排気: S_out と S_{\mathrm eff}
	•	壁物性: SEE / reflection / sticking の property maps

この層の役割は、境界がどこにあるかではなく、その境界が何をしているかを教えることです。  ￼

⸻

13. 最小実装のおすすめ

最初に実装するなら、私はこれだけに絞ります。

入力
	•	x, y または r, z
	•	d_ground, d_power, d_float, d_dielectric, d_inlet, d_outlet
	•	n_x, n_y
	•	εr
	•	Φ_bc
	•	Ebc_x, Ebc_y
	•	Q_in
	•	主要組成 1〜2 枚
	•	T_in
	•	S_out

U-Net
	•	4-level residual U-Net
	•	GroupNorm + GELU
	•	scalar condition encoder
	•	AdaGN/FiLM 的 scale-and-shift
	•	attention なし

出力
	•	φ, ne, ni, Te

loss
	•	L_field + L_wall + L_int + L_port

これでかなり強いです。
そして channel 数はまだ管理しやすく、境界数が変わる問題にも対応しやすいです。  ￼

⸻

14. 何がどう良くなるか

最後に、効果を一言で整理するとこうです。
	•	mask だけの U-Net
境界は見えるが、距離・向き・物理が見えない
	•	SDF だけ足した U-Net
境界の位置と向きは見えるが、その境界の値や役割差が弱い
	•	SDF + scalar 条件
値は分かるが、どこに効くかが弱い
	•	SDF + boundary physics maps + conditioned U-Net
「どこ」と「何をするか」が両方入るので、壁近傍・誘電体界面・入口・排気口の誤差が最も下がりやすい

今回の目的には、これがいちばんシンプルで強い構成です。  ￼

必要なら次に、これをそのまま 2D/軸対称 U-Net 用の具体チャネル定義 と 学習 loss の式 に落として書きます。