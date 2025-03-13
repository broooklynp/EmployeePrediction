import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import streamlit as st
import pandas as pd
import pickle

with open('model.pkl', 'rb') as f:
    linreg_final_2 = pickle.load(f)

st.title("Employee Monthly Salary Predictor")

years_at_company = st.number_input("Years at Company", min_value=0, max_value=20, value=1)
performance_score = st.number_input("Performance Score", min_value=1, max_value=5, value=1)
overtime_hours = st.number_input("Overtime Hours", min_value=0, max_value=30, value=0)
promotions = st.number_input("Promotions", min_value=0, max_value=3, value=0)
work_hours_per_week = st.number_input("Work Hours Per Week", min_value=0, max_value=70, value=40)
sick_days = st.number_input("Sick Days", min_value=0, max_value=15, value=0)
training_hours = st.number_input("Training Hours", min_value=0, max_value=150, value=10)
employee_satisfaction_score = st.number_input("Employee Satisfaction Score", min_value=1.00, max_value=5.00, value=1.0, step=0.1)
department = st.selectbox("Department", ["Customer Support", "Engineering", "Finance", "HR", "IT", "Legal", "Marketing", "Operations", "Sales"])
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
job_title = st.selectbox("Job Title", ["Analyst", "Consultant", "Developer", "Engineer", "Manager", "Specialist", "Technician"])
education_level = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])

if st.button("Predict"):
    input_dict = {
        'Years_At_Company': years_at_company,
        'Performance_Score': performance_score,
        'Overtime_Hours': overtime_hours,
        'Promotions': promotions,
        'Work_Hours_Per_Week': work_hours_per_week,
        'Sick_Days': sick_days,
        'Training_Hours': training_hours,
        'Employee_Satisfaction_Score': employee_satisfaction_score,
        'Department': department,
        'Gender': gender,
        'Job_Title': job_title,
        'Education_Level': education_level
    }

    input_df = pd.DataFrame([input_dict])

    pred = linreg_final_2.predict(input_df)

    st.success(f"Predicted Employee Monthly Salary: ${pred[0]:,.2f}")