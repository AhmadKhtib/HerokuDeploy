import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go


st.set_page_config(page_title='IBM Employee Attrition', page_icon='👨‍💻👩‍💻', layout="centered", initial_sidebar_state="auto")
st.title('IBM Employee Attrition App')

st.write(""" 
    App Developed by [Ahmad Alkhateeb](https://www.linkedin.com/in/ahmad-ihab-alkhatib/) \n
This app predicts the **Employee Attrition**

""")
arr = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

#############################################################################
st.markdown("")
see_data = st.expander('Click here to see the raw data 👇')
with see_data:
    st.dataframe(data=arr.reset_index(drop=True))
#########################################################
# Create an Altair # Set the color scheme
color_scheme = 'orange'  
chart = alt.Chart(arr).mark_bar(color=color_scheme).encode(
    x=alt.X('Age:Q'),
    y='count()',
    tooltip=['count()']
).properties(
    width=600,
    height=400,
    title='Distribution of employees Ages'
).interactive()
st.altair_chart(chart)
##############################################
fig = px.histogram(arr, x="Attrition", category_orders=dict(Gender=["Male", "Female"]), color='Gender',
                   title='Attrition and Gender', color_discrete_map={'Male': 'blue', 'Female': 'orange'})
st.plotly_chart(fig)
################################################
st.title('Employees Number vs Monthly Income')
fig = px.scatter(arr, x='EmployeeNumber', y='MonthlyIncome')
fig.update_xaxes(title='Employee Number')
fig.update_yaxes(title='Monthly Income')

fig.update_traces(marker=dict(color='red'))
fig.update_layout(
    dragmode='zoom',
    hovermode='closest'
)
point_size = st.slider('Point Size', min_value=1, max_value=10, value=5, step=1)
fig.update_traces(marker=dict(size=point_size))
st.plotly_chart(fig)
###############################################

fig = px.scatter(
    arr,
    x="MonthlyIncome",
    y="Age",
    size="MonthlyRate",
    color="JobRole",
    hover_name="JobRole",
    log_x=True,
    size_max=25
)

tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
with tab1:
    
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
with tab2:
    st.plotly_chart(fig, theme=None, use_container_width=True)

###############################################################

# Count the values in the 'JobSatisfaction' column
satisfaction_counts = arr['JobSatisfaction'].value_counts()
marital_status_counts = arr['MaritalStatus'].value_counts()
job_level_counts = arr['JobLevel'].value_counts()
PerformanceRating_counts = arr['PerformanceRating'].value_counts()
gender_counts = arr['Gender'].value_counts()
RelationshipSatisfaction_counts = arr['RelationshipSatisfaction'].value_counts()
WorkLifeBalance_counts = arr['WorkLifeBalance'].value_counts()
OverTime_counts = arr['OverTime'].value_counts()
JobRole_counts = arr['JobRole'].value_counts()
Department_counts = arr['Department'].value_counts()
EducationField_counts = arr['EducationField'].value_counts()
BusinessTravel_counts = arr['BusinessTravel'].value_counts()

chart_types = ['bar', 'pie', 'pie', 'pie', 'bar', 'pie', 'pie', 'bar', 'pie', 'pie', 'pie', 'bar']

st.title('Employees Attrition Dashboard')



# subplots 
cols_G01 = st.columns(2)

with cols_G01[0]:
    st.subheader('Proportion of Gender')
    fig_gender = go.Figure(go.Bar(x=gender_counts.index, y=gender_counts.values ))
    fig_gender.update_layout(height=300, width=350)
    st.plotly_chart(fig_gender)

with cols_G01[1]:
    st.subheader('Proportion of Job Satisfaction')
    fig_satisfaction = go.Figure(go.Pie(labels=satisfaction_counts.index, values=satisfaction_counts.values))
    fig_satisfaction.update_layout(height=300, width=350)
    st.plotly_chart(fig_satisfaction)

cols_G02 = st.columns(2)

with cols_G02[0]:
    st.subheader('Proportion of Marital Status')
    fig_marital = go.Figure(go.Pie(labels=marital_status_counts.index, values=marital_status_counts.values))
    fig_marital.update_layout(height=300, width=350)
    st.plotly_chart(fig_marital)

with cols_G02[1]:
     st.subheader('Proportion of Job Level')
     fig_job_level = go.Figure(go.Pie(labels=job_level_counts.index, values=job_level_counts.values))
     fig_job_level.update_layout(height=300, width=350)
     st.plotly_chart(fig_job_level)

cols_G03 = st.columns(2)

with cols_G03[0]:
     st.subheader('Proportion Performance Rating')
     fig_performance = go.Figure(go.Bar(x=PerformanceRating_counts.index, y=PerformanceRating_counts.values))
     fig_performance.update_layout(height=300, width=350)
     st.plotly_chart(fig_performance)

with cols_G03[1]:
     st.subheader('Proportion Relationship Satisfaction')
     fig_relationship = go.Figure(go.Pie(labels=RelationshipSatisfaction_counts.index, values=RelationshipSatisfaction_counts.values))
     fig_relationship.update_layout(height=300, width=350)
     st.plotly_chart(fig_relationship)

cols_G04 = st.columns(2)

with cols_G04[0]:
     st.subheader('Proportion Work Life Balance')
     fig_worklife = go.Figure(go.Pie(labels=WorkLifeBalance_counts.index, values=WorkLifeBalance_counts.values))
     fig_worklife.update_layout(height=300, width=350)
     st.plotly_chart(fig_worklife)

with cols_G04[1]:
    st.subheader('Proportion Over Time')
    fig_overtime = go.Figure(go.Bar(x=OverTime_counts.index, y=OverTime_counts.values))
    fig_overtime.update_layout(height=300, width=350)
    st.plotly_chart(fig_overtime)

cols_G05 = st.columns(2)

with cols_G05[0]:
     st.subheader('Proportion Job Role')
     fig_jobrole = go.Figure(go.Pie(labels=JobRole_counts.index, values=JobRole_counts.values))
     fig_jobrole.update_layout(height=300, width=350)
     st.plotly_chart(fig_jobrole)

with cols_G05[1]:
    st.subheader('Proportion Department')
    fig_department = go.Figure(go.Pie(labels=Department_counts.index, values=Department_counts.values))
    fig_department.update_layout(height=300, width=350)
    st.plotly_chart(fig_department)

cols_G06 = st.columns(2)

with cols_G06[0]:
     st.subheader('Proportion Education Field')
     fig_education = go.Figure(go.Pie(labels=EducationField_counts.index, values=EducationField_counts.values))
     fig_education.update_layout(height=300, width=350)
     st.plotly_chart(fig_education)

with cols_G06[1]:
    st.subheader('Proportion Business Travel')
    fig_business = go.Figure(go.Bar(x=BusinessTravel_counts.index, y=BusinessTravel_counts.values))
    fig_business.update_layout(height=300, width=350)
    st.plotly_chart(fig_business)


################################################################


def get_chart():
    
    x2 = arr['MonthlyIncome']
    x3 = arr['DailyRate']

    hist_data = [x2, x3]

    group_labels = ['Monthly Income', 'EmployeeNumber']
    colors = ['#A56CC1', '#A6ACEC', '#63F5EF']

    # Create distplot with curve_type set to 'normal'
    fig = ff.create_distplot(hist_data, group_labels, colors=colors, bin_size=100, show_rug=False)

    fig.update_layout(title_text='Distribution of Monthly Income and Employees Number')

    tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
    with tab1:
        st.plotly_chart(fig, theme="streamlit")
    with tab2:
        st.plotly_chart(fig, theme=None)

get_chart()
###############################################################
import statsmodels.api as sm
import plotly.express as px

def plt_attribute_correlation(aspect1, aspect2, color_dim):
    df_plot = df_data_filtered
    trendline = "ols" if corr_type == "Regression Plot (Recommended)" else None
    color_discrete_sequence = ['#f21111'] if color_dim in label_attr_dict_correlation.values() else None
    fig = px.scatter(df_plot, x=aspect1, y=aspect2, trendline=trendline,
                     trendline_color_override='orange',
                     #trendline=dict(line=dict(width=6), band='confidence', bandcolor='rgba(0, 0, 255, 0.2)'),
                     labels={aspect1: aspect1, aspect2: aspect2},
                     color=df_plot[color_dim],
                     color_discrete_sequence=color_discrete_sequence)
    
    fig.update_layout(
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='red', size=18),
        xaxis=dict(title=dict(text=aspect1,font=dict(color='red',size=30)),
                   tickfont=dict(color='red')),
        yaxis=dict(title=dict(text=aspect2, font=dict(color='red',size=30)),
                  tickfont=dict(color='red')),
        showlegend=True,
        
        autosize=False,
        dragmode='zoom',
        hovermode='closest',
        #hoverlabel=dict(bgcolor='lime'),
        width=700,
        height=600
    )
    
    return fig



df_data_filtered = arr  

# Mapping of attribute labels to column names
label_attr_dict_correlation = {
    'Age': 'Age',
    'DailyRate': 'DailyRate',
    'DistanceFromHome': 'DistanceFromHome',
    'EmployeeNumber': 'EmployeeNumber',
    'MonthlyIncome': 'MonthlyIncome',
    'MonthlyRate': 'MonthlyRate',
    'NumCompaniesWorked': 'NumCompaniesWorked',
    'TrainingTimesLastYear': 'TrainingTimesLastYear',
    'YearsInCurrentRole': 'YearsInCurrentRole',
    'YearsSinceLastPromotion': 'YearsSinceLastPromotion',
    'YearsWithCurrManager': 'YearsWithCurrManager',
}

Color_label_attr_dict_correlation={'Age': 'Age',
    'DailyRate': 'DailyRate',
    'DistanceFromHome': 'DistanceFromHome',
    'EmployeeNumber': 'EmployeeNumber',
    'MonthlyIncome': 'MonthlyIncome',
    'MonthlyRate': 'MonthlyRate',
    'NumCompaniesWorked': 'NumCompaniesWorked',
    'TrainingTimesLastYear': 'TrainingTimesLastYear',
    'YearsInCurrentRole': 'YearsInCurrentRole',
    'YearsSinceLastPromotion': 'YearsSinceLastPromotion',
    'YearsWithCurrManager': 'YearsWithCurrManager',
    'BusinessTravel': 'BusinessTravel',
    'Department': 'Department',
    'Education': 'Education',
    'EducationField': 'EducationField',
    'EnvironmentSatisfaction': 'EnvironmentSatisfaction',
    'Gender': 'Gender',
    'JobInvolvement': 'JobInvolvement',
    'JobLevel': 'JobLevel',
    'JobRole': 'JobRole',
    'JobSatisfaction': 'JobSatisfaction',
    'MaritalStatus': 'MaritalStatus',
    'OverTime': 'OverTime',
    'PerformanceRating': 'PerformanceRating',
    'RelationshipSatisfaction': 'RelationshipSatisfaction',
    'StockOptionLevel': 'StockOptionLevel'
}
corr_plot_types = ["Regression Plot (Recommended)", "Standard Scatter Plot"]

row11_spacer1, row11_1, row11_spacer2, row11_2, row11_spacer3 = st.columns((.2, 2.3, .4, 4.4, .2))
with row11_1:
    st.markdown("""Investigate the correlation of attributes, but keep in mind correlation does not alawys imply causation. Do Employees that are older get more Monthly Income ?""")
    corr_type = st.selectbox("What type of correlation plot do you want to see?", corr_plot_types)
    y_axis_aspect2 = st.selectbox("Which attribute do you want on the y-axis?", list(label_attr_dict_correlation.keys()))
    x_axis_aspect1 = st.selectbox("Which attribute do you want on the x-axis?", list(label_attr_dict_correlation.keys()))
    color_dim = st.selectbox("Which attribute do you want as the color dimension?", list(Color_label_attr_dict_correlation.keys()))

with row11_2:
    fig = plt_attribute_correlation(x_axis_aspect1, y_axis_aspect2, color_dim)
    st.plotly_chart(fig)

###############################################################
st.sidebar.header(' Employee Features ')
st.sidebar.subheader('\nTry to change the Features below to see what the model will predict . ⬇⬇⬇')


        # Function to handle user input features
def user_input_features():
            
            BusinessTravel = st.sidebar.selectbox('Business Travel', ('Non-Travel', 'Travel_Rarely', 'Travel_Frequently'))
            Department = st.sidebar.selectbox('Department', ('Sales', 'Research & Development', 'Human Resources'))

            Education = st.sidebar.selectbox('Education', ('Below College', 'College', 'Bachelor', 'Master', 'Doctor'))
            EducationField = st.sidebar.selectbox('Education Field', ('Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources'))
            EnvironmentSatisfaction = st.sidebar.selectbox('Environment Satisfaction', ('Low', 'Medium', 'High', 'Very High'))
            
            Gender = st.sidebar.selectbox('Gender', ('Female', 'Male'))
            JobInvolvement = st.sidebar.selectbox('Job Involvement', ('Low', 'Medium', 'High', 'Very High'))
            JobLevel = st.sidebar.selectbox('Job Level', ('Entry', 'Junior', 'Mid-Level', 'Senior', 'Executive'))
            
            JobRole = st.sidebar.selectbox('Job Role', ('Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'))        
            JobSatisfaction = st.sidebar.selectbox('Job Satisfaction', ('Low', 'Medium', 'High', 'Very High'))
            
            MaritalStatus = st.sidebar.selectbox('Marital Status', ('Single', 'Married', 'Divorced'))
            OverTime = st.sidebar.selectbox('Over Time', ('Yes', 'No'))
            PerformanceRating = st.sidebar.selectbox('Performance Rating', ('Excellent', 'Outstanding'))
            RelationshipSatisfaction = st.sidebar.selectbox('Relationship Satisfaction', ('Low', 'Medium', 'High', 'Very High'))
            
            StockOptionLevel = st.sidebar.selectbox('Stock Option Level', ('No stock options granted', 'Stock options granted at a discount', 'Stock options granted at a premium price', 'Stock options granted at market price'))

            Age = st.sidebar.slider('Age', 18, 60, 40)
            DailyRate = st.sidebar.slider('Daily Rate', 1, 15000, 300)
            DistanceFromHome = st.sidebar.slider('Distance From Home', 1, 30, 10)
            EmployeeNumber = st.sidebar.slider('Employee Number', 1, 2000, 50)
            MonthlyIncome = st.sidebar.slider('Monthly Income', 10, 20000, 500)
            MonthlyRate = st.sidebar.slider('Monthly Rate', 1, 30000, 15000)
            NumCompaniesWorked = st.sidebar.slider('Number of Companies Worked', 0, 15, 10)
            TrainingTimesLastYear = st.sidebar.slider('Training Times Last Year', 0, 10, 3)
            YearsInCurrentRole = st.sidebar.slider('Years in Current Role', 0, 30, 4)
            YearsSinceLastPromotion = st.sidebar.slider('Years Since Last Promotion', 0, 15, 4)
            YearsWithCurrManager = st.sidebar.slider('Years with Current Manager', 0, 12, 4)

            data = {
                'Age': Age,
                'DailyRate': DailyRate,
                'DistanceFromHome': DistanceFromHome,
                'EmployeeNumber': EmployeeNumber,
                'MonthlyIncome': MonthlyIncome,
                'MonthlyRate': MonthlyRate,
                'NumCompaniesWorked': NumCompaniesWorked,
                'TrainingTimesLastYear': TrainingTimesLastYear,
                'YearsInCurrentRole': YearsInCurrentRole,
                'YearsSinceLastPromotion': YearsSinceLastPromotion,
                'YearsWithCurrManager': YearsWithCurrManager,
                'BusinessTravel': BusinessTravel,
                'Department': Department,
                'Education': Education,
                'EducationField': EducationField,
                'EnvironmentSatisfaction': EnvironmentSatisfaction,
                'Gender': Gender,
                'JobInvolvement': JobInvolvement,
                'EducationField': EducationField,
                'JobLevel': JobLevel,
                'JobRole': JobRole,
                'JobSatisfaction': JobSatisfaction,
                'MaritalStatus': MaritalStatus,
                'OverTime': OverTime,
                'PerformanceRating': PerformanceRating,
                'RelationshipSatisfaction': RelationshipSatisfaction,
                'StockOptionLevel': StockOptionLevel,
                
            }

            features = pd.DataFrame([data], columns=['Age', 'DailyRate', 'DistanceFromHome',
                                                    'EmployeeNumber', 'MonthlyIncome', 'MonthlyRate',
                                                    'NumCompaniesWorked', 'TrainingTimesLastYear', 'YearsInCurrentRole',
                                                    'YearsSinceLastPromotion', 'YearsWithCurrManager', 'BusinessTravel',
                                                    'Department', 'Education', 'EducationField',
                                                    'EnvironmentSatisfaction', 'Gender', 'JobInvolvement',
                                                    'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus',
                                                    'OverTime', 'PerformanceRating', 'RelationshipSatisfaction',
                                                    'StockOptionLevel'])

            return features
input_df = user_input_features()
    




ibm_raw = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

ibm = ibm_raw.drop(columns=['Attrition', 'Over18', 'EmployeeCount', 'PercentSalaryHike','YearsAtCompany', 'TotalWorkingYears',
                             'HourlyRate','WorkLifeBalance', 'StandardHours'])

df = pd.concat([input_df, ibm], axis=0)

cols_to_encode = ['Education', 'EnvironmentSatisfaction', 'JobInvolvement','JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction','JobLevel']
encoded_X = df[cols_to_encode]

encoder = OrdinalEncoder()
encoder.fit(df[cols_to_encode])
df[cols_to_encode] = encoder.transform(df[cols_to_encode])

cols_to_onehot = ['StockOptionLevel', 'BusinessTravel', 'Department','EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
encoded_cols = pd.get_dummies(df[cols_to_onehot])
df_encoded = pd.concat([df, encoded_cols], axis=1)
df_encoded.drop(cols_to_onehot, axis=1, inplace=True)
df = df_encoded.iloc[:1]

def predict_employee_attrition(df):
    load_clf = pickle.load(open('ibm.pkl', 'rb'))
    prediction = load_clf.predict(df)
    prediction_proba = load_clf.predict_proba(df)
    return prediction, prediction_proba


####################################################################
# Display user input features
st.subheader('Current input')


#st.write('Current input.')
st.dataframe(df.style.highlight_max(color='red', axis=0), height=10, use_container_width=True)

# prediction using predict_employee_attrition function
prediction, prediction_proba = predict_employee_attrition(df)
# prediction result
st.write("""
## 

""")
st.header('**Prediction**')
st.subheader('**Note :** Try to change the Features on the sidebar **On the left ** to see the value ⬇⬇ below changes  ')

emp_att = np.array(['No', 'Yes'])

#st.write( emp_att[prediction])
st.dataframe(pd.DataFrame(emp_att[prediction]).style.highlight_max(color='red',axis=0),height=10,use_container_width=True)

# prediction probabilities on a bar chart
st.subheader('**Prediction Probabilities**')
labels = ['No', 'Yes']
proba_values = prediction_proba[0]

fig = go.Figure(data=[go.Bar(x=labels, y=proba_values, text=proba_values, textposition='auto')])
fig.update_layout(
    #title_text='Prediction Probabilities',
    xaxis_title='Employee Attrition',
    yaxis_title='Probability',
    font=dict(size=18)  # Set the font size for the text on the chart
)

fig.update_traces(texttemplate='%{text:.2f}', textfont_size=20)  # Set the font size for the bar labels

st.plotly_chart(fig, use_container_width=False)
# DataFrame from the prediction probabilities
proba_df = pd.DataFrame(prediction_proba, columns=['No', 'Yes'])

proba_df_formatted = proba_df.applymap('{:.2f}'.format)

# Apply CSS styling to the DataFrame
proba_df_styled = proba_df_formatted.style\
    .set_properties(subset=pd.IndexSlice[:, :],
                    **{'background-color': 'White'})

#st.dataframe(proba_df_styled, height=10, use_container_width=True)
#st.write(prediction_proba)
st.dataframe(proba_df_formatted.style.highlight_max(color='red',axis=0),height=10,use_container_width=True)

