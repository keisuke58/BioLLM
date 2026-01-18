# 最悪パフォーマンスクラス分析

## Geneformer

### F1下位5クラス

- **Platelet**: F1=0.000, Precision=0.000, Recall=0.000, Support=1
- **DC**: F1=0.240, Precision=0.300, Recall=0.200, Support=15
- **NK**: F1=0.375, Precision=0.462, Recall=0.316, Support=38
- **B**: F1=0.410, Precision=0.727, Recall=0.286, Support=28
- **Mono**: F1=0.414, Precision=0.522, Recall=0.343, Support=35

### 混同先Top3（F1下位5クラス）

#### B
- 正解数: 8/28
- 主な混同先:
  - T: 15回
  - NK: 2回
  - DC: 2回

#### DC
- 正解数: 3/15
- 主な混同先:
  - T: 5回
  - Mono: 5回
  - B: 1回

#### Mono
- 正解数: 12/35
- 主な混同先:
  - T: 14回
  - DC: 5回
  - NK: 4回

#### NK
- 正解数: 12/38
- 主な混同先:
  - T: 22回
  - Mono: 4回

#### Platelet
- 正解数: 0/1
- 主な混同先:
  - T: 1回

## scGPT

### F1下位5クラス

- **Platelet**: F1=0.000, Precision=0.000, Recall=0.000, Support=1
- **DC**: F1=0.667, Precision=0.750, Recall=0.600, Support=15
- **NK**: F1=0.754, Precision=0.839, Recall=0.684, Support=38
- **Mono**: F1=0.877, Precision=0.842, Recall=0.914, Support=35
- **T**: F1=0.885, Precision=0.846, Recall=0.928, Support=83

### 混同先Top3（F1下位5クラス）

#### DC
- 正解数: 9/15
- 主な混同先:
  - Mono: 6回

#### Mono
- 正解数: 32/35
- 主な混同先:
  - DC: 3回

#### NK
- 正解数: 26/38
- 主な混同先:
  - T: 12回

#### Platelet
- 正解数: 0/1
- 主な混同先:
  - T: 1回

#### T
- 正解数: 77/83
- 主な混同先:
  - NK: 5回
  - B: 1回
