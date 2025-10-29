import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from PIL import Image
import matplotlib.pyplot as plt

# load the diabetes dataset
diabetes_df = pd.read_csv('diabetes.csv')

# group the data by outcome
diabetes_mean_df = diabetes_df.groupby('Outcome').mean()

# split the data into input and target variables
X = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']

# scale input variables
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# create an SVM model
model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)

# accuracy
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)


def app():
    img = Image.open(r"img.jpeg")
    img = img.resize((200, 200))
    st.image(img, caption="Diabetes Image", width=200)

    st.title('Diabetes Prediction')

    st.sidebar.title("Input Features")
    st.sidebar.markdown("Adjust the sliders below to input patient data and predict diabetes likelihood.")

    gender = st.sidebar.radio("Gender", ["Female", "Male"])

    # Gender-dependent pregnancies
    if gender == "Male":
        preg = st.sidebar.slider('Pregnancies', 0, 17, 0, disabled=True)
    else:
        preg = st.sidebar.slider('Pregnancies', 0, 17, 3)

    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin', 0, 846, 30)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725, 0.001)
    age = st.sidebar.slider('Age', 21, 81, 29)

    input_data = [preg, glucose, bp, skinthickness, insulin, bmi, dpf, age]
    input_data_nparray = np.asarray(input_data).reshape(1, -1)

    # ✅ FIX: Scale user input before predicting
    scaled_input = scaler.transform(input_data_nparray)

    if st.button("Predict"):
        prediction = model.predict(scaled_input)
        if prediction[0] == 1:
            st.markdown("<h3 style='text-align: center; color: red;'>🩺 The person is likely diabetic.</h3>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='text-align: center; color: green;'>✅ The person is not diabetic.</h3>",
                        unsafe_allow_html=True)

    st.header('Dataset Summary')
    st.write(diabetes_df.describe())

    st.header('Distribution by Outcome')
    st.write(diabetes_mean_df)

    st.header('Model Accuracy')
    st.write(f'Train set accuracy: {train_acc:.2f}')
    st.write(f'Test set accuracy: {test_acc:.2f}')

    st.header("Glucose Level Distribution by Outcome")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(
        [diabetes_df[diabetes_df['Outcome'] == 0]['Glucose'],
         diabetes_df[diabetes_df['Outcome'] == 1]['Glucose']],
        bins=20,
        color=['green', 'red'],
        label=['Non-Diabetic', 'Diabetic'],
        alpha=0.7,
        edgecolor='black'
    )
    ax.set_title('Glucose Level Distribution')
    ax.set_xlabel('Glucose Level')
    ax.set_ylabel('Frequency')
    ax.legend()
    st.pyplot(fig)


if __name__ == '__main__':
    app()
