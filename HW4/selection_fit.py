import pandas as pd
from ELI5 import PermutationImportance
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier as RFC
rom sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, auc, \
                            log_loss, roc_auc_score, average_precision_score, confusion_matrix



dataset = pd.read_csv('dataset/train.csv', sep=';')
X = dataset.drop(['user_id', 'is_churned'], axis=1)
y = dataset['is_churned']

X_mm = MinMaxScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_mm,
                                                    y,
                                                    test_size=0.3,
                                                    shuffle=True,
                                                    stratify=y,
                                                    random_state=100)

X_train_balanced, y_train_balanced = SMOTE(random_state=42, sampling_strategy=0.5).fit_sample(X_train, y_train)

def rfc_fit_predict(X_train, y_train, X_test, y_test):
    print('Обучение модели на тренировочных данных...')
    clf = RFC(max_depth=9, n_estimators=200, random_state=100, n_jobs=4)

    clf.fit(X_train, y_train)
    return clf

def perm_imp(X_train_balanced, y_train_balanced, X_test, y_test, X):
    print('Отбор признаков...')
    fitted_clf = rfc_fit_predict(X_train_balanced, y_train_balanced, X_test, y_test)
    perm = PermutationImportance(fitted_clf, random_state=42).fit(X_train_balanced, y_train_balanced)

    res = pd.DataFrame(X.columns, columns=['feature'])
    res['score'] = perm.feature_importances_
    res['std'] = perm.feature_importances_std_
    res = res.sort_values(by='score', ascending=False).reset_index(drop=True)

    good_features = res.loc[res['score'] > 0]['feature']
    return good_features

def xgb_fit_predict(X_train, y_train, X_test, y_test):
    clf = xgb.XGBClassifier(max_depth=3,
                            n_estimators=100,
                            learning_rate=0.1,
                            nthread=5,
                            subsample=1.,
                            colsample_bytree=0.5,
                            min_child_weight = 3,
                            reg_alpha=0.,
                            reg_lambda=0.,
                            seed=42,
                            missing=1e10)

    clf.fit(X_train, y_train, eval_metric='aucpr', verbose=10)
    predict_proba_test = clf.predict_proba(X_test)
    predict_test = clf.predict(X_test)
    precision_test, recall_test, f1_test, log_loss_test, roc_auc_test = evaluation(
        y_test, predict_test, predict_proba_test[:, 1])
    return clf

# Обучаем бейзлайн
model_xgb = xgb_fit_predict(X_train_balanced, y_train_balanced, X_test, y_test)




