import joblib
import pandas as pd
import numpy as np
model=joblib.load("HR_Model.pkl")

print(model.named_steps['preprocessor'])

yeni_tahmin=pd.DataFrame(
    np.array([[35,         # Age
                          800,        # DailyRate
                          10,         # DistanceFromHome
                          3,          # Education
                          1,          # EmployeeCount
                          1024,       # EmployeeNumber
                          4,          # EnvironmentSatisfaction
                          60,         # HourlyRate
                          3,          # JobInvolvement
                          2,          # JobLevel
                          4,          # JobSatisfaction
                          7000,       # MonthlyIncome
                          20000,      # MonthlyRate
                          2,          # NumCompaniesWorked
                          15,         # PercentSalaryHike
                          3,          # PerformanceRating
                          3,          # RelationshipSatisfaction
                          1,          # StockOptionLevel
                          80,         #StandardHours
                          10,         # TotalWorkingYears
                          2,          # TrainingTimesLastYear
                          3,          # WorkLifeBalance
                          5,          # YearsAtCompany
                          3,          # YearsInCurrentRole
                          2,          # YearsSinceLastPromotion
                          4,          # YearsWithCurrManager
                          'Travel_Rarely',   # BusinessTravel
                          'Research & Development', # Department
                          'Life Sciences',   # EducationField
                          'Male',            # Gender
                          'Research Scientist', # JobRole
                          'Married',         # MaritalStatus
                          'Y',               # Over18
                          'Yes']]),
    columns=['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeCount',
       'EmployeeNumber', 'EnvironmentSatisfaction', 'HourlyRate',
       'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
       'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike',
       'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours',
       'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
       'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
       'YearsSinceLastPromotion', 'YearsWithCurrManager','BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole',
       'MaritalStatus', 'Over18', 'OverTime']
)

tahmin=model.predict(yeni_tahmin)
tahmin_label = ["Çalışan Ayrılmayacak" if i==0 else "Çalışan Ayrılır" for i in tahmin]

print("TAHMİN==",tahmin)
print("Tahmin (Öder/Ödeyemez):", tahmin_label)