import streamlit as st
import joblib
import pandas as pd


model=joblib.load("Student_Perfomance.pkl")
le=joblib.load("Label_ecoded.pkl")
ss=joblib.load("Standard_Scaler.pkl")

sidebar=st.sidebar.selectbox("Pages",["ABOUT","PREDICTION APP"])


if sidebar=="PREDICTION APP":
    st.title("Students Perfomance Prediction")
    st.write("This is an app to give a perfomance index out of 100")


    st.subheader("Enter Student details")

    hours_studied=st.number_input("Hours Studied")
    previous_scores=st.number_input("Previous Scores")
    sleep_hours=st.number_input("Hours Slept")
    sample_paper_practd=st.number_input("Sample Question Papers Practiced")
    eca=st.selectbox("Extracurricular_Activities",("Yes","No"),index=None,placeholder="Select_Option")
    
    # st.write(eca)
    if st.button("Predict"):
        eca=le.transform([[eca]])
        st.subheader("Students perfomance index score")
        data=pd.DataFrame({"Hours Studied":hours_studied,"Previous Scores":previous_scores,"Sleep Hours":sleep_hours,"Sample Question Papers Practiced":sample_paper_practd,"Extracurricular_Activities":eca})
        scaled_data=ss.transform(data)
        prediction=model.predict(scaled_data)
        if prediction[0]>60:
            st.success(round(prediction[0],2))
        elif prediction[0]>=40:
            st.warning(round(prediction[0],2))
        else:
            st.error(round(prediction[0],2))
    




if sidebar=="ABOUT":
    st.title("App Description")
    st.write("This app is designed to predict the perfomance of students out of hundred scale")
    # st.link_button("My_github", "https://github.com/naazneggar")
    st.page_link("https://github.com/naazneggar",label="My Github")


