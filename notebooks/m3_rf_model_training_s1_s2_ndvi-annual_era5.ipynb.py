import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from google.colab import files

# ==============================
# 1) CSV yükle
# ==============================
uploaded = files.upload()
fname = list(uploaded.keys())[0]
df = pd.read_csv(fname)

# --- OC temizliği: '< LOD' gibi metinleri sayısala çevir ---
df['OC_raw'] = df['OC']  # keep the original
df['OC'] = (df['OC'].astype(str)
            .str.replace(',', '.', regex=False)
            .str.strip())

# '< LOD' veya benzeri metinleri NaN yap
df.loc[df['OC'].str.contains('LOD', case=False, na=False), 'OC'] = np.nan
df.loc[df['OC'].str.contains('<', na=False), 'OC'] = np.nan

# Sayısala çevir (çevrilemeyenler NaN olur)
df['OC'] = pd.to_numeric(df['OC'], errors='coerce')

print("Rows with '<LOD' removed:", (df['OC_raw'].astype(str).str.contains('LOD', case=False, na=False)).sum())
print("OC non-numeric count (dropped later):", df['OC'].isna().sum())
print("Raw shape:", df.shape)
print(df.head())

# ==============================
# 2) Feature set
# ==============================
feature_cols = ['B2','B3','B4','B8','B11','B12','NDVI',
                'VV','VH','VV_div_VH','VV_minus_VH',
                't2m_C','tp_sum','sm1','sm2']

print(df.columns.tolist())
print("After dropna:", df.shape)

df = df.dropna(subset=feature_cols + ['OC'])

# 3) X ve y
X = df[feature_cols].astype(float)
y = np.log1p(df['OC'])


# ==============================
# 3) X ve y
# ==============================
X = df[feature_cols].astype(float)
y = df['OC'].astype(float)

print("\nOC distribution:")
print(y.describe())

# ---- Log transform toggle ----
USE_LOG1P = True
if USE_LOG1P:
    y = np.log1p(y)
    print("\nUsing log1p(OC) as target.")

# ==============================
# 4) Train / Test split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 5) Model
# ==============================
rf = RandomForestRegressor(
    n_estimators=800,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2
)
rf.fit(X_train, y_train)

# ==============================
# 6) Test metrics
# ==============================
pred = rf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("\nTest metrics")
print("Test R2:", round(r2,3))
print("Test RMSE:", round(rmse,3))

# ==============================
# 7) 5-fold CV
# ==============================
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = cross_val_score(rf, X, y, cv=cv, scoring='r2')
cv_rmse = np.sqrt(-cross_val_score(rf, X, y, cv=cv,
                                    scoring='neg_mean_squared_error'))

print("\nCV metrics (5-fold)")
print("CV R2 mean±std:", round(cv_r2.mean(),3), "±", round(cv_r2.std(),3))
print("CV RMSE mean±std:", round(cv_rmse.mean(),3), "±", round(cv_rmse.std(),3))

# ==============================
# 8) Feature importance
# ==============================
imp = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature importance")
print(imp)

imp.to_csv("feature_importance.csv", index=False)
files.download("feature_importance.csv")
