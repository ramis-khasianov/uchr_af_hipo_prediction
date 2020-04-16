import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import spearmanr

df_emp = pd.read_excel('hipo.xlsx')
df_dem = pd.read_excel('af_hipo_data.xlsx', sheet_name='hr_age_tenure')
df_af = pd.read_excel('af_hipo_data.xlsx', sheet_name='af_candidates')
df_res = pd.read_excel('af_hipo_data.xlsx', sheet_name='af_results')


# --------- Data Preprocessing -----------------

df_main = df_emp.rename(columns={
    'Главное подразделение ': 'department',
    'Структурное подразделение': 'unit',
    'Должность': 'position',
    'Руководитель': 'manager',
    'Уровень по отношению к ГД': 'position_level',
    'ФИО': 'name',
    '1 – HiPo|HiPro \n2 – Hi Potential \n3 – Hi Professional \n4 – Low performer': 'hipo_hipro_result',
    'Assess first Brain': 'brain_result',
})
df_main.loc[(df_main['HiPo'] == 1) & (df_main['HiPro'] == 1), 'HiPoPro'] = 1
df_main['HiPoPro'].fillna(0, inplace=True)
df_main['level'] = df_main['position_level'].str.split('-').str[1].astype(float)

df_dem = df_dem.rename(columns={
    'Сотрудник_УИД': 'employee_id',
    'Возраст': 'age',
    'СтажЧистый': 'tenure'
})
df_main = df_main.merge(df_dem[['employee_id', 'age', 'tenure']], how='left', on='employee_id')

df_af['first_interval'] = (df_af['shape_finished_date'] - df_af['created_at']).dt.days
df_af['second_interval'] = (df_af['drive_finished_date'] - df_af['shape_finished_date']).dt.days
df_af['third_interval'] = (df_af['brain_finished_date'] - df_af['drive_finished_date']).dt.days

df_main = df_main.merge(
    df_af[['employee_id', 'uuid', 'assessments_finished',
           'first_interval', 'second_interval', 'third_interval']],
    how='left',
    on='employee_id'
)

df_main.to_excel('final_user_list.xlsx', index=False)

df_res['title'] = df_res['id'].map('{:0>2}'.format) + '_' + df_res['name']
df_res_reg = pd.pivot_table(df_res, columns='title', index='uuid', values='score').reset_index()
df_res_norm = pd.pivot_table(df_res, columns='title', index='uuid', values='normalized_score').reset_index()

df_main = df_main.merge(df_res_norm, how='left', on='uuid')

columns_to_exclude = [
    'department', 'unit', 'position', 'manager', 'position_level',
    'name', 'hipo_hipro_result', 'brain_result', 'employee_id',  'uuid',
    'assessments_finished'
]


def my_get_dummies(df_init, col):
    df = df_init.copy()
    for val in df[col].unique()[1:]:
        df.loc[df[col] == val, val] = 1
        df[val] = df[val].fillna(0)
    return df


df = df_main[(df_main['assessments_finished'] == 3) & (df_main['hipo_hipro_result'].isin([1, 2, 3, 4, 3.5]))]
df = my_get_dummies(df, 'department')
df = df.drop(columns=columns_to_exclude)
all_features = df[[x for x in df.columns if x not in ['HiPo', 'HiPro', 'HiPoPro']]]

print(f'From {df_main.shape[0]} initial, {df.shape[0]} remained')
# rom 295 initial 159 remained

# ------- Correlations ----------


def get_correlations(all_features, target):
    df_stats = pd.DataFrame(columns=['feature', 'spearman', 'r'])
    for i in range(len(all_features.columns.values)):
        spearman, r = spearmanr(all_features.iloc[:, i], target)
        df_stats.loc[i] = [all_features.columns.values[i], spearman, r]
    df_stats.loc[df_stats['r'] < 0.05, 'relevance'] = 1
    df_stats.loc[df_stats['r'] < 0.01, 'relevance'] = 2
    df_stats.loc[df_stats['r'] < 0.001, 'relevance'] = 3
    df_stats['relevance'].fillna(0, inplace=True)

    relevance_1 = list(df_stats[df_stats['relevance'] >= 1]['feature'])
    relevance_2 = list(df_stats[df_stats['relevance'] >= 2]['feature'])
    relevance_3 = list(df_stats[df_stats['relevance'] >= 3]['feature'])
    return df_stats, {'relevance_005': relevance_1, 'relevance_001': relevance_2, 'relevance_0001': relevance_3}

df_stats_all = pd.DataFrame()
for targ in ['HiPo', 'HiPro', 'HiPoPro']:
    target = df[targ]
    df_stats_targ = get_correlations(all_features, target)[0]
    df_stats_targ['target'] = targ
    df_stats_all = pd.concat([df_stats_all, df_stats_targ], ignore_index=True)

df_stats_all.to_csv('correlations.csv', index=False)

# ------- Scaling ---------

target = df['HiPoPro']
df_stats, explored_features = get_correlations(all_features, target)

if explored_features:
    features = all_features[[x for x in all_features.columns if x in explored_features['relevance_005']]]
else:
    features = all_features.copy()

scaler = StandardScaler()
scaler.fit(features)
scaled_features = scaler.transform(features)

# ------- Logistic Regression ---------

for _ in range(10):
    x_train, x_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.30)
    regressor = LogisticRegression()
    regressor.fit(x_train, y_train)
    y_hat = regressor.predict(x_test)
    print(f'LogReg ROC_AUC: {roc_auc_score(y_test, y_hat)} Accuracy: {accuracy_score(y_test, y_hat)}')

    # Average ROC AUC = 55-60%


# --------- Support Vectors ---------

for _ in range(10):
    x_train, x_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.25)
    classifier = SVC(kernel='linear', random_state=0)
    classifier.fit(x_train, y_train)
    y_hat = classifier.predict(x_test)
    print(f'SVM ROC_AUC: {roc_auc_score(y_test, y_hat)} Accuracy: {accuracy_score(y_test, y_hat)}')

    # Average ROC AUC = 50-55%


# --------- XGBoost --------

for _ in range(2):
    x_train, x_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.25)
    classifier = GradientBoostingClassifier()
    param_grid = {
        'learning_rate': [1, 0.5, 0.1],
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'min_samples_split': [10],
        'min_samples_leaf': [5, 7],
        'max_features': [2, 4, 6]
    }

    gridsearch = GridSearchCV(classifier, param_grid, scoring='roc_auc', cv=6)
    best_model = gridsearch.fit(x_train, y_train)
    y_hat = best_model.predict(x_test)
    print(f'XGB ROC_AUC: {roc_auc_score(y_test, y_hat)} Accuracy: {accuracy_score(y_test, y_hat)}')
    print(f'XGB best params: {gridsearch.best_params_}')

    # average ROC AUC = 50-55%
