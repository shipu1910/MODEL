import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

st.title("🎓 NSTI Student Performance Prediction App")
st.write("Data Mining & Machine Learning using Decision Tree")


# Load Dataset

df = pd.read_csv("student_performance.csv")

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())


# Pattern Finding

st.subheader("📈 Pattern Finding (Grouped by Result)")
grouped_data = df.groupby("Result").mean()
st.dataframe(grouped_data)


# Prepare Data

X = df[["Attendance", "StudyHours", "PreviousMarks"]]
y = df["Result"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)


# Train Model

model = DecisionTreeClassifier()
model.fit(X_train, y_train)


# Prediction on Test Data
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.subheader("✅ Model Performance")
st.write(f"**Accuracy:** {accuracy * 100:.2f} %")

st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# New Student Prediction
st.subheader("🧑‍🎓 Predict New Student Result")

attendance = st.slider("Attendance (%)", 0, 100, 75)
study_hours = st.slider("Study Hours per day", 0, 10, 3)
previous_marks = st.slider("Previous Marks", 0, 100, 65)

new_student = [[attendance, study_hours, previous_marks]]

if st.button("Predict Result"):
    prediction = model.predict(new_student)

    if prediction[0] == 1:
        st.success("🎉 Prediction: Student PASS")
    else:
        st.error("❌ Prediction: Student FAIL")

# Visualization
st.subheader("📉 Student Performance Pattern")

fig, ax = plt.subplots()
scatter = ax.scatter(df["Attendance"], df["PreviousMarks"], c=y_encoded)
ax.set_xlabel("Attendance")
ax.set_ylabel("Previous Marks")
ax.set_title("Student Performance Pattern")

st.pyplot(fig)