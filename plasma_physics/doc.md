結論から言うと、MLP/U-Net共通で効くのは、ネットワーク内部をPINNのように全面拘束することではなく、出力場を一度ソルバー格子に落として、その上で「限定された物理」を離散残差として課すことです。PINOは「粗い教師データ + 高解像度のPDE制約」という形で精度を上げる枠組みを示し、近年のPILNOでも粗いラベルに対してより細かいコロケーション格子で物理 residual を入れると高解像度誤差が大きく改善し、PI-RINOでは物理空間に有限差分ソルバを統合してAutodiff系と同等精度で高速化できることが示されています。したがって、あなたのケースでは「全部をPINN化」するより、Poisson・連続・壁フラックス・電子エネルギー・Scharfetter–Gummel(SG)離散フラックスのような“部分物理”を、出力後の共通物理層として入れるのが最も筋が良いです。 ￼

低圧プラズマでこれが特に重要なのは、壁境界と電子運動論の省略が見かけ以上に大きな物理誤差を生むからです。RFプラズマの連続体モデルでは、電子壁フラックスの扱いを簡略化するだけで、バルク密度が約10%低下したり、20–30%増えたり、場合によっては2–3倍に跳ねたりし、純ドリフト境界では誘電体壁でシースが形成されない例まで報告されています。さらに、アルゴンICPのごく低圧例では、Maxwellian EEDF 仮定がプラズマ密度を最大93%過大評価しました。しかも別のハイブリッド研究では、低圧側ではイオンの drift-diffusion 近似自体が破綻し得ることが示されています。これらの数値は条件依存で普遍値ではありませんが、「Poisson+連続」だけでは足りず、壁境界・EEDF閉包・離散フラックスが主要誤差源になることはかなり明確です。 ￼

以下では、低温・低圧の流体/ハイブリッドモデルを前提に、MLP/U-Netの両方へ共通実装しやすい制約を整理します。連続体モデルとしては、**LFA（local field approximation）と、追加で電子エネルギー式を解くLMEA（local mean energy approximation）**が代表的で、どちらもPoisson方程式を併用します。 ￼

記号は、n_s: 種密度、\Gamma_s: 粒子フラックス、\phi: 電位、E=-\nabla\phi、\bar\varepsilon_e or T_e: 平均電子エネルギー/電子温度、\sigma_w: 壁面電荷です。

1. 制約候補となる理論式の整理

理論式	一般式	内容	有効な物理値	主な効果	根拠
Poisson 方程式	\nabla\!\cdot(\varepsilon\nabla\phi)=-\rho_c=-e(\sum_i Z_i n_i-n_e)	空間電荷から電位・電場を決める中核式	\phi, E, n_e, n_i, シース位置	空間電荷と電位の結合を強制し、シース位置・自己バイアス・電場分布を安定化	￼
種連続の式	\partial_t n_s+\nabla\!\cdot\Gamma_s=S_s	各種粒子の局所保存	n_e, n_i, 励起種, 負イオン	局所保存、生成消滅と輸送の整合、偽の密度源/吸い込みの抑制	￼
Drift–diffusion フラックス	\Gamma_e=-\mu_e n_eE-D_e\nabla n_e,\;\Gamma_i=Z_i\mu_i n_iE-D_i\nabla n_i（符号は定義依存）	電場駆動と拡散の輸送則	\Gamma_e,\Gamma_i,J,n_s	密度勾配と電場に対する輸送方向を合わせ、局所の輸送整合を改善	￼
電子エネルギー保存式 / LMEA	\partial_t(n_e\bar\varepsilon_e)+\nabla\!\cdot\xi_e=E\!\cdot\!J_e-n_e\sum_r K_r\Delta\varepsilon_r	Joule加熱、熱伝導、非弾性損失を扱う	T_e,\bar\varepsilon_e, 反応率, イオン化源	反応源・発光/励起・電力吸収の整合が改善	￼
EEDF / Boltzmann 閉包	\mu_e,D_e,k_r=\mathcal{G}(\bar\varepsilon_e \text{ or } E/N) または EBE/Boltzmann 由来	電子輸送係数・反応率を電子運動論に結び付ける	\mu_e,D_e,k_r,S_s,S_\omega	低圧での非Maxwell性・非局所性を反映し、密度と電力吸収の誤差を抑える	￼
電流連続 / 誘電体壁電荷進化	\partial_t\rho_c+\nabla\!\cdot J=0, \partial_t\sigma_w=e(\Gamma_i-\Gamma_e)\!\cdot n, -E\!\cdot n=\sigma_w/\varepsilon_0	壁面へ流入する電流が表面電荷と境界電場を決める	\sigma_w, E_n, \phi, 自己バイアス	誘電体壁近傍の電場・壁電荷・シース応答を改善	￼
バルク準中性 + 両極性拡散	\sum_i Z_i n_i \approx n_e（バルク）, J_e\approx J_i	シース外のバルクでは準中性近似が有効	n_e,n_i, バルク電位, バルク密度	バルク内部の不自然な空間電荷を抑え、滑らかな内部場を得やすい	￼
Bohm 判定 + 壁電位 / 壁フラックス	u_i\ge c_s=\sqrt{k_BT_e/m_i}, \Phi_w\approx-(k_BT_e/e)\ln\sqrt{m_i/(2\pi m_e)}（浮遊壁の代表式）	シース入口条件、壁電位、壁への粒子流束	シース端, 壁フラックス, イオン入射エネルギー	シース幅・壁フラックス・自己バイアス・イオンエネルギーの物理性を強化	￼
Scharfetter–Gummel 離散フラックス	F_{K\sigma}=\tau_\sigma[B(-\Delta\psi)n_K-B(\Delta\psi)n_\sigma]	対流拡散支配での面フラックスの指数適合離散化	面フラックス, 急峻勾配, シース境界層	数値振動を減らし、定常解保存・散逸性・急峻シースの再現を改善	￼
半陰的 Poisson 更新 / 電荷補正式	\rho^{m+1}\approx \rho^m-\Delta t\,\nabla\!\cdot J^{m+1}, \nabla\!\cdot(\varepsilon E^{m+1})=\rho^{m+1}	時間離散とPoissonを一貫させる solver 理論	過渡 E,\rho_c,\phi	時間積分時の位相ずれ・電場符号反転・大ステップ不安定を抑える	￼
大域粒子/電力バランス	\frac{d}{dt}\int_\Omega n_s dV+\oint_{\partial\Omega}\Gamma_s\!\cdot n\,dA=\int_\Omega S_s dV, \int_\Omega E\!\cdot J_e dV=P_{\rm loss}+...	連続式・エネルギー式の体積積分形	総電子数, 総イオン数, 吸収電力	局所誤差は残っても、総量のドリフトやバイアスを抑える	連続式・エネルギー式の積分形として導入可。 ￼

この表のうち、最初に入れる価値が高いのは、Poisson、連続、壁フラックス/Bohm、SG、電子エネルギー/EEDFです。逆に、準中性を全領域にかけるのは危険で、シースを潰しやすいです。シースはDebye長スケール、presheath はイオン平均自由行程スケールで、そもそもバルクと同じ損失で縛るべきではありません。 ￼

2. PINO 粒度での入れ方：共通の設計原理

PINO的にまとめるなら、損失は

\mathcal{L}
=
\lambda_d \mathcal{L}_{data}^{(h_c)}
+
\sum_j \lambda_j \mathcal{L}_{phys,j}^{(h_f)}
+
\lambda_{bc}\mathcal{L}_{bc}^{(h_f)}
+
\lambda_{glob}\mathcal{L}_{global}

です。ここで h_c は教師データ格子、h_f はそれより細かい物理格子で、粗い教師 + 細かい物理 loss にします。これはPINOやPILNOの考え方にほぼ一致します。 ￼

実装上の肝は、Autodiffの連続PDE residual にこだわらず、ソルバーと同じ有限差分/有限体積/SGの離散残差を使うことです。そうすると、U-Netでは出力テンソルにそのまま stencil を当てればよく、MLPでも同じ格子点でサンプリングして同じ stencil を当てられるので、物理層を共通化できます。PI-RINOはこの「物理空間で有限差分ソルバを統合する」方向が有効であることを示しています。 ￼

さらに、低圧プラズマではbulk / sheath / wall / high-Péclet を分けて損失をかけるのが重要です。例えば
m_{\rm bulk}=1 if |\rho_c|/(e n_e)<\tau_q、
m_{\rm sheath}=1-m_{\rm bulk}、
m_{\rm Pe}=1 if Pe_{\rm face}>Pe_{th}
のようなマスクを用意し、準中性は bulk のみ、Bohm と壁フラックスは sheath/wall のみ、SG は high-Péclet 面のみに入れます。これは “限定物理” を入れるという要求に非常に合っています。

3. 理論式ごとの「学習への入れ方」と効果

理論式	学習の制約の入れ方	組み込む際の工夫	学習面での効果	精度面での効果	根拠
Poisson	L_P=\| \nabla\!\cdot(\varepsilon\nabla\phi)+e(\sum Z_in_i-n_e)\|_2^2	\phi は基準電位を1点固定するか平均ゼロで gauge fix。U-Net/MLPとも fine grid で評価	\phi-n の結合を早期に学習しやすい	空間電荷と電位の整合、シース位置の改善	￼
種連続	L_C=\|\partial_t n_s+\nabla\!\cdot\Gamma_s-S_s\|_2^2	正値性のため n_s=\mathrm{softplus}(z_s) か \log n_s 出力にする。局所 loss に加えて体積積分 loss を足す	偽の局所 source/sink に逃げにくくなる	総粒子数のドリフト減少、局所密度分布の改善	￼
Drift–diffusion フラックス	L_{DD}=\|\Gamma_s-\Gamma_s^{DD}(n_s,E,T)\|_2^2	密度 head とは別に face-flux head を持たせると安定。弱衝突低圧では重みを下げる	輸送方向と大きさの自由度を縮める	勾配の向き、拡散/ドリフト比の改善	￼
電子エネルギー式	L_\varepsilon=\|\partial_t(n_e\bar\varepsilon_e)+\nabla\!\cdot\xi_e-E\!\cdot\!J_e+n_e\sum K_r\Delta\varepsilon_r\|_2^2	最初から強くかけず、Poisson/continuity が安定した後に weight を上げる	源項の stiff さで学習が崩れるのを避けやすい	T_e, ionization zone, absorbed power の再現向上	￼
EEDF/Boltzmann 閉包	\bar\varepsilon_e を補助 head で出し、\mu_e,D_e,k_r を微分可能 table/小NN で計算して整合 loss	BOLSIG+ 等で作った係数表を emulator 化。未ラベル条件にも physics-only で適用	低データでも化学・輸送の自己整合を保ちやすい	低圧での密度/電力吸収/反応率の破綻を抑える	￼
壁電荷・電流連続	L_{wall}=\|\partial_t\sigma_w-e(\Gamma_i-\Gamma_e)\!\cdot n\|^2+\|-E\!\cdot n-\sigma_w/\varepsilon_0\|^2	境界セルだけで評価。金属壁と誘電体壁で loss を分ける	境界条件学習を局所化でき、内部場の学習を邪魔しにくい	自己バイアス、壁近傍電場、シース応答の改善	￼
準中性 / 両極性	L_{qn}=\|m_{bulk}(\sum Z_in_i-n_e)\|_2^2 + 必要なら J_e\approx J_i	全領域にかけない。sheath mask を必ず入れる	バルク内部の無意味な荷電揺らぎを減らす	内部の電位・密度分布を滑らかにする	￼
Bohm / 壁フラックス	L_B=\|m_{sheath}\,\mathrm{ReLU}(c_s-u_i)\|_2^2 + 完全壁フラックス残差	sheath edge を |\rho_c|/(en_e) や |E| の閾値で近似。純ドリフト壁BCだけは避ける	壁近傍の学習を physics-guided にできる	シース幅、壁粒子束、イオン入射エネルギーの改善	￼
SG 離散フラックス	L_{SG}=\sum_{\text{faces}}\|\Gamma_{face}-\Gamma_{SG}\|_2^2	中央差分でなく solver と同じ face operator を使う。high-Péclet 面だけに絞る	高勾配域でも loss が安定しやすい	数値振動・ringing を減らし、急峻シース再現が向上	￼
半陰的 Poisson 更新	L_{imp}=\|\nabla\!\cdot(\varepsilon E^{m+1})-[\rho^m-\Delta t\nabla\!\cdot J^{m+1}]\|_2^2	時系列モデルや1-step predictorで有効。初期時刻側をやや重視する temporal weighting が効く	大きい \Delta t でも学習が不安定になりにくい	位相遅れ、電場反転、過渡の非物理振動を抑制	￼
大域粒子/電力バランス	L_{glob}=\sum_s|\oint\Gamma_s\!\cdot n-\int S_s|^2+|P_{abs}-P_{loss}|^2	batch 単位で体積積分。weight は弱めでよい	低コストで OOD 時の発散を抑えやすい	総密度・総電力の systematic bias を補正	￼

ここでの重要点は3つです。

1つ目は、壁境界は “残差の端” ではなく独立の主損失として扱うことです。電子境界条件の近似の違いだけでバルク密度が大きく変わるので、ここを軽く扱うと全体が崩れます。 ￼

2つ目は、低圧では電子エネルギー/EEDFを省略しないことです。Poisson と continuity が合っていても、\mu_e,D_e,k_r が誤っていれば、見かけ上もっともらしい n_e,\phi が出ていても、反応源・電力吸収・シース応答は間違います。アルゴンICPの very low pressure 例では、その影響が極めて大きいことが示されています。 ￼

3つ目は、弱衝突低圧では drift-diffusion を盲信しないことです。イオン輸送の慣性が効くなら、DD 残差は強拘束にせず、重みを落とすか、full ion fluid / kinetic teacher 由来の補助損失に置き換える方が安全です。 ￼

4. どの組み合わせが有用か

組み合わせ	含める理論式	向いている状況	有用な理由	注意点
最小で効く共通構成	Poisson + 種連続 + 完全壁フラックス + SG + 大域粒子バランス	定常の \phi,n_e,n_i 空間分布	電位-密度結合、局所保存、壁感度、離散輸送を最小セットで押さえられる。まずこれが一番費用対効果が高い	低圧で T_e や反応率が重要なら次の構成へ
低圧標準構成	最小構成 + 電子エネルギー式 + EEDF/Boltzmann閉包	ICP/CCP の低圧、反応源や power deposition を出したい場合	低圧では非Maxwellian EEDF が密度や電力吸収を大きく動かすため、ここを入れないと“見た目だけ合う”危険が高い	実装が重いので、まず Boltzmann 係数表の emulator 化から始めるのが現実的
bulk–sheath 分離構成	Poisson + bulk準中性 + Bohm + 壁電荷 + 完全壁フラックス	シース幅、壁束、自己バイアスが重要な場合	bulk では準中性、sheath では Bohm/壁条件、と役割分担できるので損失衝突が減る	sheath mask の定義が必要。準中性を全領域に入れない
時間依存/RF 構成	最小構成 + 半陰的Poisson更新 + 壁電荷進化 + 大域電力バランス	RF波形、自己バイアス、時系列 surrogate	時間離散と電荷保存を一貫させると位相ずれや非物理振動が減る。PILNO の temporal weighting と相性が良い	前時刻状態が必要。sequence model か one-step recurrent が前提
弱衝突・より低圧補強構成	低圧標準構成 + イオン慣性/ion flux 方程式 or kinetic teacher loss	低圧RF、イオン位相や壁入射エネルギーまで重視	DD の破綻領域を避けられる。壁イオン束や IEDF の一貫性が上がる	実装コスト上昇。全部に使う必要はない

私の見立てでは、最も有用な組み合わせは次の順です。

第1候補は
Poisson + 種連続 + 完全壁フラックス + SG
です。理由は、今の「物理的に整合しない」問題の大半が、電位-密度の非整合、局所保存の破れ、壁/シースの崩れ、急峻勾配での離散化不整合に起因するからです。これは MLP/U-Net のどちらにも共通で効きます。 ￼

第2候補は
第1候補 + 電子エネルギー式 + EEDF/Boltzmann閉包
です。低圧ではここが一気に重要になります。特に ICP/CCP で source term や power deposition まで合わないなら、この追加が最も効く可能性が高いです。 ￼

第3候補は
bulk準中性 + Bohm + 壁電荷
を、領域マスク付きで入れる構成です。これはシースの見え方、自己バイアス、壁束が重要なときに効きますが、全領域一括で入れると逆効果になりやすいので、最初から全面適用しない方が安全です。 ￼

5. 実装するなら、私はこう進めます

1段階目は、出力を solver-native 変数に寄せることです。
最終出力は少なくとも \phi,\log n_e,\log n_i を持たせ、可能なら補助 head として \bar\varepsilon_e も出します。U-NetでもMLPでも、最終的には同じ格子上へ評価して physics layer に渡します。

2段階目は、粗い教師データ + 細かい物理 loss に変えることです。
これはPINО/PILNOの流儀そのままで、教師はシミュレータの標準保存間隔、物理 loss はその2倍〜4倍細かい格子で計算します。未ラベル条件に対しても physics-only で回せます。PILNO は virtual inputs を使って OOD 側の物理 supervision を増やしています。 ￼

3段階目は、壁とシースを独立管理することです。
bulk には準中性、wall/sheath には Bohm・壁電荷・完全壁フラックスを入れます。ここを分けないと、bulk を良くする損失が sheath を潰し、その逆も起きます。 ￼

4段階目は、低圧なら EEDF 閉包を追加することです。
最も現実的なのは、BOLSIG+ 等で (E/N) または \bar\varepsilon_e から \mu_e,D_e,k_r を返す表を作り、それを小さな differentiable emulator にする方法です。これなら full Boltzmann solver を学習ループに直接入れなくてもよいです。 ￼

要するに、あなたのケースで本当に効くのは、「全方程式PINN」ではなく、出力後に置く“離散・領域分離・壁重視”の物理層です。
最初の一手としては、Poisson + 連続 + 完全壁フラックス + SG。
低圧でさらに伸ばすなら、電子エネルギー + EEDF閉包。
シース/壁を詰めるなら、Bohm + 壁電荷 + bulk準中性マスク。
この順が、精度と実装コストのバランスが最も良いと思います。