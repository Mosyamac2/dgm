import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Load the data
data = """
id,id_object,book_number,rooms,amount,prepay,price_per_night,pmnt_type,date_book,date_cancel,date_in,nights,date_out,channel,status_book,object,guests,room_type_agg
1,1,20200108-6634-58579847,1,11590.0,11590.0,5795.0,Банк. карта: Банк Россия (банк. карта),2019-12-31,,2020-01-08,2,2020-01-10,Сайт,Активный,Игора,2,Стандарт
2,1,20200213-6634-58575019,1,24605.0,24605.0,8201.67,Банк. карта: Банк Россия (банк. карта),2019-12-31,,2020-02-13,3,2020-02-16,Сайт,Активный,Игора,2,Стандарт
3,1,20200101-6634-58569957,2,30600.0,0.0,15300.0,Гарантия банковской картой,2019-12-31,,2020-01-01,1,2020-01-02,OTA,Активный,Игора,2,Стандарт
... (rest of the data) ...
224,1,20191210-6634-57212994,1,6400.0,6400.0,6400.0,Банк. карта: Банк Россия (банк. карта),2019-12-08,,2019-12-10,1,2019-12-11,Сайт,Активный,Игора,1,Стандарт
"""

# Read CSV data
df = pd.read_csv(pd.compat.StringIO(data))

# Preprocessing
## Convert dates and calculate features
date_cols = ['date_book', 'date_cancel', 'date_in', 'date_out']
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

## Create target variable: 1 = not canceled, 0 = canceled
df['target'] = (df['status_book'] == 'Активный').astype(int)

## Create new features
df['prepay_ratio'] = np.where(df['amount'] > 0, df['prepay'] / df['amount'], 0)
df['lead_time'] = (df['date_in'] - df['date_book']).dt.days
df['is_direct'] = (df['channel'] == 'Сайт').astype(int)

## Handle payment type categories
df['is_bank_card'] = df['pmnt_type'].str.contains('Банк. карта', na=False).astype(int)
df['is_guarantee'] = df['pmnt_type'].str.contains('Гарантия', na=False).astype(int)
df['is_external'] = df['pmnt_type'].str.contains('Внешняя', na=False).astype(int)

# Select features
features = [
    'prepay_ratio', 'lead_time', 'is_direct',
    'is_bank_card', 'is_guarantee', 'is_external',
    'nights', 'guests', 'rooms', 'price_per_night'
]
target = 'target'

# Split data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing pipeline
num_features = ['prepay_ratio', 'lead_time', 'nights', 'guests', 'rooms', 'price_per_night']
cat_features = ['is_direct', 'is_bank_card', 'is_guarantee', 'is_external']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', make_pipeline(
            SimpleImputer(strategy='median'),
            num_features
        ),
        ('cat', make_pipeline(
            SimpleImputer(strategy='most_frequent'),
            cat_features
        )
    ]
)

# Create and train model
model = make_pipeline(
    preprocessor,
    StandardScaler(),
    RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
)

model.fit(X_train, y_train)

# Evaluate
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"Model AUC: {auc:.4f}")

# Example prediction
sample = X_test.iloc[[0]]
prob_not_cancel = model.predict_proba(sample)[0][1]
print(f"\nSample prediction probability (not cancel): {prob_not_cancel:.4f}")
print(f"Actual value: {y_test.iloc[0]}")
print("Sample features:")
print(sample)