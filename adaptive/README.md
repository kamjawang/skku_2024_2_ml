# Boosting Fair Classifier Generalization through Adaptive Priority Reweighing 관련 정리
## 01. 논문 관련
- 논문 : Boosting Fair Classifier Generalization through Adaptive Priority Reweighing https://arxiv.org/abs/2309.08375
- 참고 코드 : [https://github.com/pyg-team/pytorch-frame](https://github.com/che2198/APW)
- dataset : Adult (https://archive.ics.uci.edu/dataset/2/adult)
- task : Classification



## 02. 코드 관련
- load_data.py
  1) csv 데이터 로드
  2) 전처리 - one-hot encoding, categorical encoding, StandardScaler
  3) 결측치 0으로 처리
 
- demographic_parity.py
  1) Demographic Parity (인구 통계학적 동등성): 예측이 민감한 속성에 독립적임을 보장
  2) ![image](https://github.com/user-attachments/assets/b12bf90a-b916-4397-8ea9-8bd82ddf55a7)

- equal_opportunity.py
  1) Equal Opportunity (동등한 기회): 민감한 속성 값에 관계없이 양성 샘플에 대한 참 양성 비율(TPR)이 같도록 함.
  2) ![image](https://github.com/user-attachments/assets/881fd32d-ff74-4410-8343-9401b56b2030)

- equalized_odds.py
  1) Equalized Odds (균등한 확률): 민감한 속성 값에 관계없이 TPR과 FPR이 같도록 함.
  2)![image](https://github.com/user-attachments/assets/0fbb6707-de15-48ea-b214-50002005e7c0)

- 공통모듈
- evaluation.py : equalized_odds_difference 와 demographic_parity_difference 실제 계산
- def compute_group_weights : 일반적인 목적함수에 민감변수 가중치 추가하여 그륩별 가중치 업데이트
- def compute_sample_weights : 경계와의 거리를 고려하여 그륩내 샘플 가중치를 업데이트

     


