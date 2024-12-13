# Pytorch_Frame 관련 정리

## 01. 논문 관련

- 논문 : [PyTorch Frame: A Modular Framework for Multi-Modal Tabular Learning](https://arxiv.org/abs/2404.00776)
- 참고 코드 : https://github.com/pyg-team/pytorch-frame
- dataset : Adult (https://archive.ics.uci.edu/dataset/2/adult)
- task : Classification



## 02. 코드 관련

#### [01_Pytorch_Frame_Claude.ipynb](https://github.com/kamjawang/skku_2024_2_ml/blob/main/pytorch_frame/01_Pytorch_Frame_Claude.ipynb)

- 설명
  - 논문 기반으로 코드 작성 수행 (FTTransformer)
  - AUROC : 0.7208



#### [02_Pytorch_Frame_Claude_revision.ipynb](https://github.com/kamjawang/skku_2024_2_ml/blob/main/pytorch_frame/02_Pytorch_Frame_Claude_revision.ipynb)

- 설명
  - code 01에서 columnTransformer를 활용한 전처리 진행 시, feature가 줄어드는 오류
  - 전처리 시 코드 수정
  - AUROC : 0.9004



#### [03_Pytorch_Frame_Claude_revision_one_by_one.ipynb](https://github.com/kamjawang/skku_2024_2_ml/blob/main/pytorch_frame/03_Pytorch_Frame_Claude_revision_one_by_one.ipynb)

- 설명
  - code 02에 대한 세부적인 내용을 팀원과 공유하기 위해 함수별 분할 작성
  - 시각화 내용 추가
  - AUROC : 0.9028



#### 04_Pytorch_fairness_auroc.ipynb

- 설명
  - fairness 지표와 pytorch frame 모델 결합 시도
  - evaluation code 오류로 삭제 처리



#### [05_Pytorch_fairness_interaction_dimension_variation.ipynb](https://github.com/kamjawang/skku_2024_2_ml/blob/main/pytorch_frame/05_Pytorch_fairness_interaction_dimension_variation.ipynb)

- 설명
  - [Boosting Fair Classifier Generalization through Adaptive Priority Reweighing](https://github.com/kamjawang/skku_2024_2_ml/tree/main/adaptive) 를 참고하여 fairness 지표 추가

  - 이때 column간 상호정보량 dimension을 조정하며, 성능 변화를 파악

  - interaction_dim이라는 변수가 조정

  - AUROC 평가 미진행 / 공정성 지표만 파악

    



#### [07_Pytorch_fairness_weights_selection.ipynb](https://github.com/kamjawang/skku_2024_2_ml/blob/main/pytorch_frame/07_Pytorch_fairness_weights_selection.ipynb)

- 설명
  - [Boosting Fair Classifier Generalization through Adaptive Priority Reweighing](https://github.com/kamjawang/skku_2024_2_ml/tree/main/adaptive) 를 참고하여 fairness 지표 추가

  - 손실함수를 조정하며, 성능 변화를 파악한 코드

  - lambda_fairness 라는 변수가 조정

  - AUROC 평가 미진행 / 공정성 지표만 파악

    



#### [07_Pytorch_fairness_weights_selection.ipynb](https://github.com/kamjawang/skku_2024_2_ml/blob/main/pytorch_frame/07_Pytorch_fairness_weights_selection.ipynb)

- 설명
  - 기본 code에서 eop, op, dp 가중치를 주어 변화 파악
  - AUROC: 0.895
  - EOP : 0.0161 (기존 논문 : 0.08)
    - 가중치 : 0.5(EOP) / 0.25(EO) / 0.25(DP)

