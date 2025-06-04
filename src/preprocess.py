from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo 

def load_data():
    statlog_german_credit_data = fetch_ucirepo(id=144) 
    
    X = statlog_german_credit_data.data.features
    y = statlog_german_credit_data.data.targets

    column_names = [
        'Status', 'Duration', 'CreditHistory', 'Purpose', 'CreditAmount',
        'Savings', 'EmploymentSince', 'InstallmentRate', 'PersonalStatusSex',
        'OtherDebtors', 'ResidenceSince', 'Property', 'Age',
        'OtherInstallmentPlans', 'Housing', 'ExistingCredits', 'Job',
        'LiablePeople', 'Telephone', 'ForeignWorker'
    ]

    X.columns = column_names

    for col in X.select_dtypes(include='object').columns:
        X.loc[:, col] = LabelEncoder().fit_transform(X[col])

    y = y.replace({1: 1, 2: 0})

    return X, y