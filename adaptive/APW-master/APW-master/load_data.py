import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Define the column names for the Adult dataset
COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
           'occupation', 'relationship', 'race', 'gender', 'capital-gain',
           'capital-loss', 'hours-per-week', 'native-country', 'income']

def load_dataset():

    # Load the data, drop unnecessary columns and missing values
    ### 성준님 데이터
    all_data = pd.read_csv('./adult.csv',na_values=[' ?']).drop(['fnlwgt'], axis=1).dropna()
    # split = 0.8
    # split_idx = int(split*len(all_data))

    # 데이터 분할
    dataset_train, dataset_test = train_test_split(all_data, test_size=0.2, random_state=15)


    # dataset_train = all_data.loc[:split_idx]
    # dataset_test = all_data.loc[split_idx:].reset_index(drop=True)



    print("### DATA LOAD ADULT CSV ###")


    ### 원래 논문 데이터
    # dataset_train = pd.read_csv('./adult.data', names=COLUMNS, na_values=[' ?']).drop(['fnlwgt'], axis=1).dropna()
    # dataset_test = pd.read_csv('./adult.test', names=COLUMNS, na_values=[' ?']).drop(['fnlwgt'], axis=1).dropna()

    # Preprocess the training data
    features_train = pd.get_dummies(dataset_train.drop(['income'], axis=1))
    labels_train = pd.Categorical(dataset_train['income']).codes


    sensitive_attribute = 'gender'
    protected_attribute_train = pd.Categorical(dataset_train[sensitive_attribute]).codes.astype('float32')



    # Preprocess the test data
    features_test = pd.get_dummies(dataset_test.drop(['income'], axis=1))
    labels_test = pd.Categorical(dataset_test['income']).codes
    protected_attribute_test = pd.Categorical(dataset_test[sensitive_attribute]).codes.astype('float32')


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

    return features_train, features_test, labels_train, labels_test, protected_attribute_train, protected_attribute_test