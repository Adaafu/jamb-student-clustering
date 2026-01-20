import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="JAMB Student Cluster",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model_and_preprocessor():
    model = joblib.load("model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_model_and_preprocessor()

st.title("JAMB Students Result Cluster")

st.markdown("""
This app groups students into clusters based on:  
• **Study habits** (hours per week, assignments, attendance)  
• **School environment** (type, location, teacher quality)  
• **Family & support background** (socioeconomic status, parent involvement, education, IT knowledge, extra tutorials)

Select from the characteristics below to see which cluster best matches the profile.
""")

st.markdown("---")

cluster_descriptions = {
    0: "Low Effort Group: "
       "Study_Hours_Per_Week = 10.52, "
       "Assignments_Completed = 1.10, "
       "Attendance_Rate = 75.77%, "
       "Teacher_Quality = 2.25, ",

    1: "Moderate Effort Group: "
       "Study_Hours_Per_Week = 25.59, "
       "Assignments_Completed = 2.33, "
       "Attendance_Rate = 76.21%, "
       "Teacher_Quality = 2.09, ",

    2: "High Committment Group: "
       "Study_Hours_Per_Week = 31.14, "
       "Assignments_Completed = 3.42, "
       "Attendance_Rate = 88.95%, "
       "Teacher_Quality = 2.87, ",

    3: "Higher Teacher-Supported Group: "
       "Study_Hours_Per_Week = 18.36, "
       "Assignments_Completed = 1.40, "
       "Attendance_Rate = 88.86%, "
       "Teacher_Quality = 3.51, ",

    4: "High Attendance Group: "
       "Study_Hours_Per_Week = 17.01, "
       "Assignments_Completed = 1.36, "
       "Attendance_Rate = 91.29%, "
       "Teacher_Quality = 1.63, "
}

st.subheader("Study & School Factors")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider("Study Hours Per Week", 0, 40, 20)
    attendance = st.slider("Attendance Rate (%)", 50, 100, 85)
    teacher_quality = st.slider("Teacher Quality (1–5)", 1, 5, 3)

with col2:
    assignments = st.slider("Assignments Completed", 1, 5, 2)
    school_type = st.selectbox("School Type", ["Public", "Private"])
    school_location = st.selectbox("School Location", ["Urban", "Rural"])

st.subheader("Family & Support Factors")

col3, col4 = st.columns(2)

with col3:
    extra_tutorials = st.selectbox("Extra Tutorials", ["Yes", "No"])
    it_knowledge = st.selectbox("IT Knowledge", ["Low", "Medium", "High"])
    socioeconomic = st.selectbox("Socioeconomic Status", ["Low", "Medium", "High"])

with col4:
    parent_involvement = st.selectbox("Parent Involvement", ["Low", "Medium", "High"])
    parent_education = st.selectbox(
        "Parent Education Level",
        ["None", "Primary", "Secondary", "Tertiary"]
    )

input_df = pd.DataFrame({
    'Study_Hours_Per_Week':     [study_hours],
    'Attendance_Rate':          [attendance],
    'Teacher_Quality':          [teacher_quality],
    'Assignments_Completed':    [assignments],
    'School_Type':              [school_type],
    'School_Location':          [school_location],
    'Extra_Tutorials':          [extra_tutorials],
    'Socioeconomic_Status':     [socioeconomic],
    'IT_Knowledge':             [it_knowledge],
    'Parent_Education_Level':   [parent_education],
    'Parent_Involvement':       [parent_involvement]
})


if st.button("Find Cluster", type="primary"):
    try:
        X_input = preprocessor.transform(input_df)
        cluster = model.predict(X_input)[0]
        
        desc = cluster_descriptions.get(cluster, f"Cluster {cluster} – no description available")
        
        st.success(f"**Predicted Cluster: {cluster}**")
        st.markdown(f"**{desc}**")
        
        with st.expander("Input values sent to the model"):
            st.json(input_df.to_dict(orient='records')[0])
            
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")


st.markdown("---")
st.caption("Built by Adaafu• January 2026")
st.caption("Clustering model trained with K-Means • Data-driven student profiles")