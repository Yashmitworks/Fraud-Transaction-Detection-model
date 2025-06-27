# Enhanced Fraud Detection with Scenario Injection
import pandas as pd
import numpy as np
import zipfile
import io
import warnings
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ------------------------------
# 1. Load all .pkl files from ZIP
# ------------------------------
zip_path = "dataset.zip"
dataframes = []

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    pkl_files = [f for f in zip_ref.namelist() if f.endswith('.pkl')]
    if not pkl_files:
        raise FileNotFoundError("No .pkl files found in dataset.zip")

    for pkl_file in pkl_files:
        print(f"\U0001F4E6 Loading: {pkl_file}")
        with zip_ref.open(pkl_file) as f:
            df_part = pd.read_pickle(io.BytesIO(f.read()))
            dataframes.append(df_part)

# ------------------------------
# 2. Merge or Concatenate
# ------------------------------
try:
    from functools import reduce
    df = reduce(lambda left, right: pd.merge(left, right, on='TRANSACTION_ID', how='outer'), dataframes)
    print("✅ Merged on TRANSACTION_ID.")
except Exception:
    print("⚠️ Could not merge. Concatenating vertically...")
    df = pd.concat(dataframes, axis=0, ignore_index=True)

print(f"✅ Final shape: {df.shape}")

# ------------------------------
# 3. Basic Cleaning
# ------------------------------
df.drop_duplicates(inplace=True)
df.fillna(0, inplace=True)

# Ensure required columns exist
required_columns = {'TX_AMOUNT', 'TX_TIME_DAYS', 'CUSTOMER_ID', 'TERMINAL_ID'}
if not required_columns.issubset(df.columns):
    raise ValueError(f"Missing one of required columns: {required_columns - set(df.columns)}")

# ------------------------------
# 4. Inject Fraud Scenarios
# ------------------------------
# Rule 1: TX_AMOUNT > 220 is fraud
df.loc[df['TX_AMOUNT'] > 220, 'TX_FRAUD'] = 1

# Rule 2: Two terminals selected per day, fraud for 28 days
for day in range(df['TX_TIME_DAYS'].min(), df['TX_TIME_DAYS'].max() - 28):
    terminals = df[df['TX_TIME_DAYS'] == day]['TERMINAL_ID'].dropna().unique()
    if len(terminals) >= 2:
        selected = np.random.RandomState(seed=day).choice(terminals, size=2, replace=False)
        days_range = range(day, day + 28)
        df.loc[
            (df['TERMINAL_ID'].isin(selected)) &
            (df['TX_TIME_DAYS'].isin(days_range)), 'TX_FRAUD'
        ] = 1

# Rule 3: 3 customers per day, 1/3 of transactions for next 14 days are 5x amount + fraud
for day in range(df['TX_TIME_DAYS'].min(), df['TX_TIME_DAYS'].max() - 14):
    customers = df[df['TX_TIME_DAYS'] == day]['CUSTOMER_ID'].dropna().unique()
    if len(customers) >= 3:
        selected = np.random.RandomState(seed=day).choice(customers, size=3, replace=False)
        for cust in selected:
            mask = (df['CUSTOMER_ID'] == cust) & (df['TX_TIME_DAYS'].between(day, day + 13))
            tx_ids = df[mask].sample(frac=1/3, random_state=day).index
            df.loc[tx_ids, 'TX_AMOUNT'] *= 5
            df.loc[tx_ids, 'TX_FRAUD'] = 1

# Fill NaNs in TX_FRAUD if any
df['TX_FRAUD'].fillna(0, inplace=True)

# ------------------------------
# 5. Encode Categorical Columns
# ------------------------------
categorical_cols = df.select_dtypes(include='object').columns.tolist()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# ------------------------------
# 6. Prepare Features and Target
# ------------------------------
target = 'TX_FRAUD'
feature_columns = [col for col in df.columns if col not in [target, 'TRANSACTION_ID']]
datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
feature_columns = [col for col in feature_columns if col not in datetime_cols]
X = df[feature_columns]
y = df[target]

# ------------------------------
# 7. Train/Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------
# 8. Apply SMOTE
# ------------------------------
print(f"🧪 Before SMOTE: {np.bincount(y_train.astype(int))}")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"✅ After SMOTE: {np.bincount(y_train_resampled.astype(int))}")

# ------------------------------
# 9. Scale Features
# ------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# 10. Train XGBoost Classifier
# ------------------------------
model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    use_label_encoder=False
)
model.fit(X_train_scaled, y_train_resampled)

# ------------------------------
# 11. Evaluate Model
# ------------------------------
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
print("📌 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"\n✅ Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"🔁 ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")

# ------------------------------
# 12. Sample Predictions
# ------------------------------
sample = X_test.copy()
sample['Actual'] = y_test.values
sample['Predicted'] = y_pred
sample['Fraud Probability'] = y_prob

print("\n🔍 Sample predictions:")
print(sample.head())
print("\n🚨 Sample actual fraud cases:")
print(sample[sample['Actual'] == 1].head())

# ------------------------------
# 13. Save Model and Scaler
# ------------------------------
joblib.dump(model, 'fraud_xgboost_model.pkl')
joblib.dump(scaler, 'fraud_scaler.pkl')
print("\n💾 Model and scaler saved successfully.")