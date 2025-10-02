import pandas as pd

data=pd.read_csv("C:/Users/90507/Desktop/archive/HR-Employee-Attrition.csv")
print(data.info())
print(data.isnull().sum())

import matplotlib.pyplot as plt
import seaborn as sns

# Yaşa göre ayrılma oranı
sns.histplot(data=data, x="Age", hue="Attrition", bins=30, multiple="stack")
plt.title("Yaşa Göre Çalışan Ayrılma Durumu")
plt.show()

# Aylık maaşa göre ayrılma oranı
sns.boxplot(data=data, x="Attrition", y="MonthlyIncome")
plt.title("Aylık Maaş vs. Ayrılma Durumu")
plt.show()

# Fazla mesai (OverTime) ile ayrılma ilişkisi
sns.countplot(data=data, x="OverTime", hue="Attrition")
plt.title("Fazla Mesai ve Ayrılma Oranı")
plt.show()


X=data.drop("Attrition",axis=1)
y=data["Attrition"].map({"No":0, "Yes":1})

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=0)

num_col=X.select_dtypes(include=["float64","int64"]).columns
cat_col=X.select_dtypes(include=["object"]).columns

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
preprocessor=ColumnTransformer(
    transformers=[
        ('num',StandardScaler(),num_col),
        ('cat',OneHotEncoder(handle_unknown="ignore"),cat_col)
    ]
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier

models={
    "Logistic Regression":LogisticRegression(max_iter=1000),
    "Random Forest":RandomForestClassifier(),
    "Gradient Boosting":GradientBoostingClassifier(),
    "XGBoost":XGBClassifier()
}

from sklearn.metrics import classification_report,accuracy_score
from sklearn.pipeline import Pipeline
results={}
for name,model in models.items():
    pipeline=Pipeline(steps=[
        ('preprocessor',preprocessor),
        ('classifier',model)
    ])
    pipeline.fit(X_train,y_train)
    tahmin=pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, tahmin)

    results[name] = {"Özet Accuracy": accuracy}

    print(f"==== {name} ====")
    print("Classification Report:\n", classification_report(y_test, tahmin))

results_df = pd.DataFrame(results).T
print(results_df)

pipeline=Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('model',LogisticRegression(max_iter=1000))
])

param_grid={
    "model__C": [0.01, 0.1, 1, 10],
    "model__penalty": ["l2"],
    "model__solver": ["lbfgs", "liblinear"]
}

from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(pipeline,param_grid,cv=5,n_jobs=-1,verbose=2)

grid.fit(X_train,y_train)

print("En İyi Parametreler",grid.best_params_)
print("En İyi Skor",grid.best_score_)

import joblib
joblib.dump(grid.best_estimator_,"HR_Model.pkl")
print("Model Kaydedildi")