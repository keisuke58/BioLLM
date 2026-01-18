# 最悪パフォーマンスクラス分析

## Geneformer

### F1下位5クラス

- **NK**: F1=0.521, Precision=0.510, Recall=0.532, Support=94
- **DC**: F1=0.528, Precision=0.655, Recall=0.442, Support=43
- **B**: F1=0.635, Precision=0.741, Recall=0.556, Support=72
- **Mono**: F1=0.636, Precision=0.696, Recall=0.585, Support=94
- **T**: F1=0.723, Precision=0.665, Recall=0.791, Support=234

### 混同先Top3（F1下位5クラス）

#### B
- 正解数: 40/72
- 主な混同先:
  - T: 17回
  - NK: 12回
  - DC: 2回

#### DC
- 正解数: 19/43
- 主な混同先:
  - Mono: 12回
  - T: 11回
  - B: 1回

#### Mono
- 正解数: 55/94
- 主な混同先:
  - T: 27回
  - NK: 5回
  - DC: 4回

#### NK
- 正解数: 50/94
- 主な混同先:
  - T: 38回
  - DC: 3回
  - Mono: 2回

#### T
- 正解数: 185/234
- 主な混同先:
  - NK: 31回
  - B: 9回
  - Mono: 8回

## scGPT

### F1下位5クラス

- **Platelet**: F1=0.667, Precision=0.667, Recall=0.667, Support=3
- **DC**: F1=0.861, Precision=0.944, Recall=0.791, Support=43
- **NK**: F1=0.869, Precision=0.938, Recall=0.809, Support=94
- **Mono**: F1=0.937, Precision=0.927, Recall=0.947, Support=94
- **T**: F1=0.939, Precision=0.898, Recall=0.983, Support=234

### 混同先Top3（F1下位5クラス）

#### DC
- 正解数: 34/43
- 主な混同先:
  - Mono: 7回
  - T: 1回
  - Platelet: 1回

#### Mono
- 正解数: 89/94
- 主な混同先:
  - T: 3回
  - DC: 2回

#### NK
- 正解数: 76/94
- 主な混同先:
  - T: 18回

#### Platelet
- 正解数: 2/3
- 主な混同先:
  - NK: 1回

#### T
- 正解数: 230/234
- 主な混同先:
  - NK: 4回
