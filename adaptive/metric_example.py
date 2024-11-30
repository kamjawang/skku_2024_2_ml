
'''
    - 기계학습특론 : 공정성 지표 metric example 코드
    - 공정성 지표 :
       1) Equal Opportunity (EOP)
       2) Equalized Odds (EO)
       3) Demographic Parity (DP)

    - Input :
       - prediction : 예측값
       - true_label : 실측값
       - sensitive_feature (= protected_attribute_train / test )
       - target_label : label 선택한 값 ( 1 , 0 )

    - Output :
       - 3 가지 지표 : 퍼센트값 (%)
        1) Equal Opportunity (EOP)
        2) Equalized Odds (EO)
        3) Demographic Parity (DP)
'''

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


'''
    평가함수
'''


def equalized_odds_difference(predictions, true_labels, sensitive_features, target_label):

    binary_predictions = np.where(predictions > 0.5, 1, 0)
    positive_mask = true_labels == target_label

    tpr_0 = np.mean(binary_predictions[(sensitive_features == 0) & positive_mask] == target_label)
    tpr_1 = np.mean(binary_predictions[(sensitive_features == 1) & positive_mask] == target_label)

    return np.abs(tpr_0 - tpr_1)

def demographic_parity_difference(predictions, sensitive_features):

    binary_predictions = np.where(predictions > 0.5, 1, 0)

    rate_0 = np.mean(binary_predictions[sensitive_features == 0])
    rate_1 = np.mean(binary_predictions[sensitive_features == 1])

    return np.abs(rate_0 - rate_1)







#### 사용예시
## 1) data load
all_data = pd.read_csv('./adult.csv',na_values=[' ?']).drop(['fnlwgt'], axis=1).dropna()
split = 0.8
split_idx = int(split*len(all_data))

dataset_train = all_data.loc[:split_idx]
dataset_test = all_data.loc[split_idx:].reset_index(drop=True)

print("### DATA LOAD ADULT CSV ###")


## 2) 지표 산출에 필요한 input 인자 생성 -> 'gender'

#### 2-1) 민감한 속성
sensitive_attribute = 'gender'
protected_attribute_train = pd.Categorical(dataset_train[sensitive_attribute]).codes.astype('float32')
protected_attribute_test = pd.Categorical(dataset_test[sensitive_attribute]).codes.astype('float32')


#### 2-2) 실측값 , 예측값
features_train = pd.get_dummies(dataset_train.drop(['income'], axis=1))
labels_train = pd.Categorical(dataset_train['income']).codes

features_test = pd.get_dummies(dataset_test.drop(['income'], axis=1))
labels_test = pd.Categorical(dataset_test['income']).codes

# Align columns in test data with training data
for column in features_train.columns:
    if column not in features_test.columns:
        features_test[column] = 0

# Ensure same column order
features_test = features_test[features_train.columns]

# Standardize the data using only training data statistics
scaler = StandardScaler()
scaler.fit(features_train)

features_train = scaler.transform(features_train)
features_test = scaler.transform(features_test)




### 간이 모델
classifier = LogisticRegression(max_iter=1000)
classifier.fit(features_train, labels_train)

print("### MODEL TRAIN COMPLETE ###")


## 예측값
predictions_test = classifier.predict(features_test)
epoch = 0




delta_equal_opportunity = equalized_odds_difference(predictions_test, labels_test, protected_attribute_test,
                                                    target_label=1)

delta_equalized_odds_negative = equalized_odds_difference(predictions_test, labels_test,
                                                          protected_attribute_test, target_label=0)

delta_demographic_parity = demographic_parity_difference(predictions_test, protected_attribute_test)

delta_equalized_odds = max(delta_equal_opportunity, delta_equalized_odds_negative)




log_message = (f'Epoch {epoch + 1}\t delta_equalized_odds {delta_equalized_odds:.2%}\t delta_equal_opportunity {delta_equal_opportunity:.2%}'f'\t delta_demographic_parity {delta_demographic_parity:.2%}\t')



print(log_message)

