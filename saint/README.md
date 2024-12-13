# SAINT 관련 정리

## 01. 논문 관련

- 논문 : SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training   (https://arxiv.org/abs/2106.01342)
- 참고 코드 : [https://github.com/pyg-team/pytorch-frame](https://github.com/somepago/saint)
- dataset : Adult (https://archive.ics.uci.edu/dataset/2/adult)
- task : Classification



## 02. 코드 관련

#### SAINT_Group10_Final_논문재현.ipynb

- 설명
  - 논문에 활용된 데이터 셋을 활용하여 재현
  - AUROC : 0.9183



#### SAINT_Group10_Final.ipynb

- 설명
  - 그룹에서 통일한 Adult 데이터셋 활용
  - 전처리시 데이터에 맞게 차원수정
  - 역전파 방지

  - 이때 column간 상호정보량을 확인하며 하이퍼 파라미터 수정 및 epoch 조정

- 모델 평가 결과:
  - 정확도 (Accuracy): 0.8585
  - AUROC: 0.9129
  - 정밀도 (Precision): 0.7673
  - 재현율 (Recall): 0.6110
  - F1 점수 (F1-Score): 0.6803

- 공정성 지표:
  - Demographic Parity (DP): 0.1668
  - Equal Opportunity (EO): 0.0503
  - Equality of Odds (EOP): 0.1173
