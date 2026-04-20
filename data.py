import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib


df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)




services = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
df['Total_Services'] = df[services].apply(lambda x: (x == 'Yes').sum(), axis=1)


df['Has_Family'] = ((df['Partner'] == 'Yes') | (df['Dependents'] == 'Yes')).astype(int)


df['Avg_Monthly'] = df['TotalCharges'] / (df['tenure'] + 1)
df['Bill_Spike'] = (df['MonthlyCharges'] > df['Avg_Monthly']).astype(int)


df['High_Risk_Profile'] = ((df['Contract'] == 'Month-to-month') &
                           (df['PaymentMethod'] == 'Electronic check')).astype(int)




df = df.drop('customerID', axis=1)

le_dict = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le


X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Model Doğruluğu (Accuracy):", accuracy_score(y_test, y_pred))
print("\nSınıflandırma Raporu:\n", classification_report(y_test, y_pred))


joblib.dump(model, 'churn_model.pkl')
joblib.dump(list(X.columns), 'model_columns.pkl')
