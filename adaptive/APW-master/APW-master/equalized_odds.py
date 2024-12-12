import argparse
import numpy as np
import logging
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from load_data import load_dataset
from evaluation import *
from sklearn.metrics import roc_auc_score
import pdb


def set_seed(seed=0):
    """Set seed for reproducibility."""
    np.random.seed(seed)


def initialize_logger(args):
    """Initialize the logger for experiment tracking."""
    log_format = '%(asctime)s %(name)s:%(levelname)s:  %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


    name_manual = '1208_EQULIZED_ODDS'

    log_file = f'{args.save_dir}/{name_manual}_log.txt'



    # log_file = f'{args.save_dir}/{args.save_name}_log.txt'
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    logger.info('Arguments')
    for arg in vars(args):
        logger.info(f'{arg}: {getattr(args, arg)}')

    return logger

def compute_sample_weights(true_labels, predicted_probabilities, protected_attributes, multipliers, eta, decision_boundary=0.5):
    num_samples = len(true_labels)
    exponential_term = np.exp(-eta * abs(predicted_probabilities - decision_boundary))
    weight_component = np.zeros(num_samples)

    for attr in protected_attributes:
        weight_component += exponential_term * np.sum(attr) / np.sum(exponential_term * attr) * attr

    combined_weights = np.zeros(num_samples)
    for i, multiplier in enumerate(multipliers):
        combined_weights += multiplier * protected_attributes[i]
    
    sample_weights = combined_weights * weight_component

    return sample_weights

def compute_group_weights(predictions, true_labels, protected_attributes, alpha):
    group_weights = []

    for p in protected_attributes:
        protected_positive_idxs = np.where(np.logical_and(p > 0, true_labels > 0))
        protected_negative_idxs = np.where(np.logical_and(p > 0, true_labels <= 0))
        all_positive_idxs = np.where(true_labels > 0)
        all_negative_idxs = np.where(true_labels <= 0)
        
        weight1 = (np.sum(true_labels[protected_positive_idxs]) * np.sum(predictions[all_positive_idxs]) + alpha) / (
                np.sum(true_labels[all_positive_idxs]) * np.sum(predictions[protected_positive_idxs]) + alpha)
        
        weight2 = (np.sum(1 - true_labels[protected_negative_idxs]) * np.sum(1 - predictions[all_negative_idxs]) + alpha) / (
                np.sum(1 - true_labels[all_negative_idxs]) * np.sum(1 - predictions[protected_negative_idxs]) + alpha)

        group_weights.extend([weight1, weight2])

    return group_weights


def main(args):
    """Main function for the fairness experiment."""
    logger = initialize_logger(args)
    features_train, features_test, labels_train, labels_test, protected_attribute_train, protected_attribute_test = load_dataset()

    protected_attributes_list = [protected_attribute_train, 1 - protected_attribute_train]
    label_combinations = [protected_attribute_train * labels_train, protected_attribute_train * (1 - labels_train), \
        (1 - protected_attribute_train) * labels_train, (1 - protected_attribute_train) * (1 - labels_train)]
    fairness_multipliers = np.ones(len(label_combinations))
    sample_weights = np.array([1] * features_train.shape[0])


    print(f"alpha : {args.alpha} , eta : {args.eta}")

    # 최고 성능을 저장하기 위한 변수 초기화
    best_test_accuracy = 0
    best_test_auroc = 0
    best_delta_equalized_odds = 0
    best_delta_equal_opportunity = 0
    best_delta_demographic_parity = 0

    best_epoch = 0
    best_log_message = ""



    for epoch in range(args.epoch):
        # Train logistic regression model
        classifier = LogisticRegression(max_iter=10000)
        classifier.fit(features_train, labels_train, sample_weights)
        
        predictions_train = classifier.predict(features_train)
        prediction_probabilities = classifier.predict_proba(features_train)[:, 1].astype(np.float32)
        prediction_probabilities_test = classifier.predict_proba(features_test)[:, 1].astype(np.float32)

        # Compute weights and multipliers
        sample_weights = compute_sample_weights(labels_train, prediction_probabilities, label_combinations, fairness_multipliers, args.eta)
        group_fairness_weights = compute_group_weights(predictions_train, labels_train, protected_attributes_list, args.alpha)
        fairness_multipliers *= np.array(group_fairness_weights)

        # Evaluate model on test dataset
        predictions_test = np.squeeze(classifier.predict(features_test))
        test_accuracy = accuracy_score(labels_test, predictions_test)

        # AUROC evaluation
        test_auroc = roc_auc_score(labels_test, prediction_probabilities_test)
        
        delta_equal_opportunity = equalized_odds_difference(predictions_test, labels_test, protected_attribute_test, target_label=1)
        delta_equalized_odds_negative = equalized_odds_difference(predictions_test, labels_test, protected_attribute_test, target_label=0)
        delta_demographic_parity = demographic_parity_difference(predictions_test, protected_attribute_test)

        delta_equalized_odds = max(delta_equal_opportunity, delta_equalized_odds_negative)

        # Log results
        log_message = (f'Epoch {epoch + 1}\t delta_equalized_odds {delta_equalized_odds:.2%}\t delta_equal_opportunity {delta_equal_opportunity:.2%}'
                       f'\t delta_demographic_parity {delta_demographic_parity:.2%}\t test_accuracy {test_accuracy:.2%}\t test_auroc {test_auroc:.3} ')
        
        logger.info(log_message)
        print(log_message)


        # 최고 성능 저장
        if test_accuracy > best_test_accuracy or (test_accuracy == best_test_accuracy and delta_equalized_odds < best_delta_equalized_odds):
            best_test_accuracy = test_accuracy
            best_test_auroc = test_auroc
            best_epoch = epoch + 1
            best_log_message = log_message



    # 최고 성능 결과 출력 및 저장
    logger.info("\nBest Model Performance:")
    logger.info(f"Epoch {best_epoch}: {best_log_message}")
    print("\nBest Model Performance:")
    print(f"Epoch {best_epoch}: {best_log_message}")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--alpha', type=int, default=0)
    parser.add_argument('--eta', type=float, default=0)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--save-dir', type=str, default='./temp')
    parser.add_argument('--save-name', type=str, default='temp-name')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    set_seed()

    args.alpha = 0.1
    args.eta = 1


    try:
        main(args)
    except Exception as e:
        logging.exception('Unexpected exception! %s', e)