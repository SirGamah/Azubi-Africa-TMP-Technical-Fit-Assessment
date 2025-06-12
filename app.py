import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
#from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from io import BytesIO
from streamlit_option_menu import option_menu

# Load data
df = pd.read_csv("bank-additional-full.csv", sep=';')

# Set page config
st.set_page_config(page_title="💰 Term Deposit Subscription Predictor")

# Navigation menu
selected = option_menu(
    menu_title="💰 Term Deposit Subscription Predictor", 
    options=["Overview", "Analysis", "Train Model", "Make Prediction", "About"],
    icons=["house", "bar-chart", "cpu", "check-circle", "info-circle"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

# Overview Page
if selected == "Overview":
    # Sidebar instructions
    st.sidebar.markdown("""
    **Overview Page Instructions**  
    - Read through the description of the data.  
    - Understand the business goal: predicting term deposit subscriptions.

    You can check out the Google Colab notebook [here](https://colab.research.google.com/drive/1O2286FFEV44nIdV2qpwY9yp0k9hsd3X-?usp=sharing) for more details on the **Data Exploration** and other intuitions.
    """)

    st.title("Term Deposit Subscription Predictor")
    # Display dataset preview and overview
    st.markdown("### Dataset Preview")
    st.dataframe(df.head())

    st.markdown("### Dataset Overview")

    st.markdown("#### 📌 Client Personal and Socioeconomic Attributes")
    st.markdown("""
            <table>
            <thead>
            <tr>
            <th>Column</th>
            <th>Description</th>
            <th>Business Insight</th>
            </tr>
            </thead>
            <tbody>
            <tr>
            <td><code>age</code></td>
            <td>Client's age (numeric).</td>
            <td>Different age groups may exhibit different investment behaviors.</td> 
            </tr>
            <tr>
            <td><code>job</code></td>
            <td>Type of job (e.g., admin., technician, retired, etc).</td>
            <td>Reflects income stability and investment likelihood. Certain jobs may be more likely to subscribe.</td> 
            </tr>
            <tr>
            <td><code>marital</code></td>
            <td>Marital status (married, single, divorced, unknown).</td>
            <td>Affects financial decisions. For instance, single clients might be more flexible with investments.</td> 
            </tr>
            <tr>
            <td><code>education</code></td>
            <td>Education level (e.g., basic.4y, high.school, university.degree, etc).</td>
            <td>Often correlates with financial literacy and investment readiness.</td> 
            </tr>
            <tr>
            <td><code>default</code></td>
            <td>Has credit in default? (yes, no, unknown)</td>
            <td>Indicates credit risk. Defaulting clients may be less likely to invest.</td> 
            </tr>
            <tr>
            <td><code>housing</code></td>
            <td>Has a housing loan?</td>
            <td>May reflect financial obligations, affecting ability to invest.</td> 
            </tr>
            <tr>
            <td><code>loan</code></td>
            <td>Has a personal loan?</td>
            <td>Similar to <code>housing</code>; affects available income for investment.</td> 
            </tr>
            </tbody>
            </table>
            """, unsafe_allow_html=True)
    
    st.markdown("#### 📞 Last Contact Communication Information")
    st.markdown("""
            <table>
            <thead>
            <tr>
            <th>Column</th>
            <th>Description</th>
            <th>Business Insight</th>
            </tr>
            </thead>
            <tbody>
            <tr>
            <td><code>contact</code></td>
            <td>Communication type (cellular, telephone).</td>
            <td>Some methods may be more effective in converting clients.</td> 
            </tr>
            <tr>
            <td><code>month</code></td>
            <td>Last contact month (e.g., may, jul, nov).</td>
            <td>Subscription likelihood may vary by month due to seasonality or campaigns.</td> 
            </tr>
            <tr>
            <td><code>day_of_week</code></td>
            <td>Day of last contact.</td>
            <td>May correlate with client availability or mood.</td> 
            </tr>
            </tbody>
            </table>
            """, unsafe_allow_html=True)
    
    st.markdown("#### ⏳ Campaign-related Features")
    st.markdown("""
            <table>
            <thead>
            <tr>
            <th>Column</th>
            <th>Description</th>
            <th>Business Insight</th>
            </tr>
            </thead>
            <tbody>
            <tr>
            <td><code>campaign</code></td>
            <td>Number of contacts during the current campaign (numeric, includes last contact).</td>
            <td>Over-contacting may reduce effectiveness. High values might indicate resistance.</td> 
            </tr>
            <tr>
            <td><code>pdays</code></td>
            <td>Days since last contact from a previous campaign (`999` means never contacted).</td>
            <td>Recently contacted clients may have higher awareness.</td> 
            </tr>
            <tr>
            <td><code>previous</code></td>
            <td>Number of contacts before the current campaign.</td>
            <td>Shows client history with past campaigns.</td> 
            </tr>
            <tr>
            <td><code>poutcome</code></td>
            <td>Outcome of previous marketing campaign (e.g., success, failure, nonexistent).</td>
            <td>Strong indicator. A past success often correlates with future success.</td> 
            </tr>
            </tbody>
            </table>
            """, unsafe_allow_html=True)
    
    st.markdown("#### 💰 Economic Context Attributes")
    st.markdown("""
            <table>
            <thead>
            <tr>
            <th>Column</th>
            <th>Description</th>
            <th>Business Insight</th>
            </tr>
            </thead>
            <tbody>
            <tr>
            <td><code>emp.var.rate</code></td>
            <td>Employment variation rate (quarterly indicator).</td>
            <td>Macroeconomic variable — can affect clients' financial outlook.</td> 
            </tr>
            <tr>
            <td><code>cons.price.idx</code></td>
            <td>Consumer price index (monthly).</td>
            <td>Reflects inflation; high inflation may deter investments.</td> 
            </tr>
            <tr>
            <td><code>cons.conf.idx</code></td>
            <td>Consumer confidence index (monthly).</td>
            <td>High confidence → higher likelihood of investments.</td> 
            </tr>
            <tr>
            <td><code>euribor3m</code></td>
            <td>3-month Euribor rate (interest rate benchmark).</td>
            <td>Influences attractiveness of bank deposits.</td> 
            </tr>
            <tr>
            <td><code>nr.employed</code></td>
            <td>Number of employees (proxy for job market conditions).</td>
            <td>More employment may increase overall income and deposits.</td> 
            </tr>
            </tbody>
            </table>
            """, unsafe_allow_html=True)
    
    st.markdown("""
            <table>
            <thead>
            <tr>
            <th>Column</th>
            <th>Description</th>
            <th>Business Insight</th>
            </tr>
            </thead>
            <tbody>
            <tr>
            <td><code>y</code></td>
            <td>Client subscribed to term deposit? (yes or no)</td>
            <td>Target variable for prediction</td> 
            </tr>
            </tbody>
            </table>
            """, unsafe_allow_html=True)
    
    st.warning("NOTE: The column descriptions were **NOT** provided with the dataset but were inferred using ChatGPT.")


    st.markdown("### Dataset Statistics")
    st.write("Number of rows:", df.shape[0])
    st.write("Number of columns:", df.shape[1])

    st.markdown("#### 📊 EDA Results")

    st.markdown("##### ✅ Missing Values")
    st.write("**None**. All 41,188 entries are complete—no missing values in any column.")

    st.markdown("##### 🎯 Target Variable (`y`) Distribution")
    st.markdown("""
    <ul>
    <li><strong>No:</strong> 4,640 (11.3%)</li>
    <li><strong>No:</strong> 36,548 (88.7%)</li>
    </ul>
    """, unsafe_allow_html=True)

    # Plot target variable distribution
    target_counts = df['y'].value_counts().reset_index()
    target_counts.columns = ['Subscription', 'Count'] 

    # Plot
    fig = px.bar(
        target_counts,
        x='Subscription',
        y='Count',
        color='Subscription',
        title='Target Variable Distribution (Subscribed to Term Deposit)',
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig.update_layout(
        xaxis_title='Subscription',
        yaxis_title='Count',
        title_x=0.5
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### ➡️ Class Imbalance is significant")
    st.write("The dataset is imbalanced, with a significant majority of clients not subscribing to term deposits. This will need to be addressed during model training.")

    st.markdown("##### 📈 Initial Observations from Numerical Features")
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    numerical_summary = df[numerical_features].describe().T
    numerical_summary['range'] = numerical_summary['max'] - numerical_summary['min']
    numerical_summary['variance'] = numerical_summary['std'] ** 2
    numerical_summary = numerical_summary[['mean', 'std', 'min', 'max', 'range', 'variance']]
    st.dataframe(numerical_summary, use_container_width=True)

    st.markdown("##### 📊 Categorical Features Overview")
    st.dataframe(df.describe(include=['object']), use_container_width=True)

    st.markdown("##### 🔍 Correlation Matrix")
    # Prepare data
    df_corr = df.copy()
    df_corr['y'] = df_corr['y'].map({'no': 0, 'yes': 1})
    corr_matrix = df_corr.select_dtypes(include=np.number).corr().round(2)
    corr_melted = corr_matrix.reset_index().melt(id_vars='index')
    corr_melted.columns = ['Feature1', 'Feature2', 'Correlation']
    fig = px.imshow(
        corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale='RdBu',
        zmin=-1, zmax=1,
        labels=dict(x="Features", y="Features", color="Correlation"),
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap (Numerical Features)"
    )

    fig.update_layout(title_x=0.5)

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""**Notable high correlations**:
                <ul>
                <li><code>euribor3m</code>, <code>nr.employed</code> and <code>emp.var.rate</code> are highly correlated.</li>
                <li><code>duration</code> has a strong positive correlation with the target (<code>y</code>) but must be dropped before modeling as it is only known after the contact.</li>
                </ul>
                """, unsafe_allow_html=True)   

    # Features to visualize
    features = ['age', 'campaign', 'pdays', 'previous', 'euribor3m', 'nr.employed']

    st.subheader("📈 Distribution of Numerical Features by Subscription Outcome")

    # Show each histogram in expandable sections
    for feature in features:
        with st.expander(f"📊 Distribution of {feature.title()} by Subscription"):
            fig = px.histogram(
                df,
                x=feature,
                color='y',
                barmode='stack',
                nbins=30,
                title=f"Distribution of {feature.title()} by Subscription",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(
                xaxis_title=feature.title(),
                yaxis_title="Count",
                title_x=0.5
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("""**Observations**:
        <ul>
        <li>Clients with fewer previous contacts (<code>campaign</code>) and higher <code>euribor3m</code> rates appear more likely to subscribe.</li>
        <li><code>pdays</code> is mostly <code>999</code>, meaning most clients were not contacted previously.</li>
        </ul>""", unsafe_allow_html=True)

    # List of categorical features
    categorical_features = ['job', 'marital', 'education', 'default', 'housing',
                            'loan', 'contact', 'month', 'day_of_week', 'poutcome']

    st.subheader("📊 Distribution of Categorical Features by Subscription Outcome")

    # Option 1: Using Expanders
    for feature in categorical_features:
        with st.expander(f"📊 {feature.title()} vs Subscription"):
            fig = px.histogram(
                df,
                x=feature,
                color='y',
                barmode='group',  # or 'stack' if you prefer
                title=f"{feature.title()} vs Subscription",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(
                xaxis_title=feature.title(),
                yaxis_title="Count",
                xaxis_tickangle=-45,
                title_x=0.5
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""**Observations**:       
        <ul>
        <li>Features like <code>contact</code>, <code>month</code>, <code>poutcome</code>, and <code>job</code> show noticeable variation in subscription rates.</li>
        <li><Example: clients contacted via <code>cellular</code> and those with successful previous outcomes (<code>poutcome</code>) are more likely to subscribe.</li>
        </ul>""", unsafe_allow_html=True)

    
    #st.subheader("📦 Boxplots for Outlier Detection")
    #tabs = st.tabs([f"{feat.title()}" for feat in features])

    #for tab, feature in zip(tabs, features):
        #with tab:
            #fig = px.box(
                #df,
                #x='y',
                #y=feature,
                #color='y',
                #points='all',
                #title=f"{feature.title()} by Subscription",
                #color_discrete_sequence=px.colors.qualitative.Set2
            #)
            #fig.update_layout(
                #xaxis_title="Subscription",
                #yaxis_title=feature.title(),
                #title_x=0.5
            #)
            #st.plotly_chart(fig, use_container_width=True)
    #st.markdown("""**Observations**:
        #<ul>
        #<li>Boxplots reveal outliers in features like <code>age</code>, <code>campaign</code>, and <code>pdays</code>.</li>
        #<li>Outliers in <code>age</code> may represent older clients who are less likely to subscribe.</li>
        #<li><code>pdays</code> shows many clients with no previous contact, indicating a potential area for improvement in outreach strategies.</li>
        #<li>Distributions are skewed; normalization or transformation may help.</li>
        #</ul>""", unsafe_allow_html=True)
    st.markdown("### Further Observations")
    st.markdown("""**Observations**:
        <ul>
        <li>Some columns like <code>job</code>, <code>marital</code>, <code>education</code>, etc have "<code>unknown</code>" as values. This might not necassary need any cleaning as removing them would reduce the data size</li>
        <li><code>jan - for January</code> and <code>feb - February</code> are missing from the <code>month</code> column in the dataset.</li>
        <li><code>sat - for Sarturday</code> and <code>sun - for Sunday</code> are missing from the <code>day_of_week</code> column of the dataset. This might be because these are weekend days.</li>
        </ul>""", unsafe_allow_html=True)
    

# Analysis Page
elif selected == "Analysis":
    # Sidebar instructions
    st.sidebar.markdown("""
    **Analysis Page Instructions**  
    - Use the dropdown to select a category.  
    - Visualizations will update based on your selection.
    """)
    # Analysis section
    st.title("Exploratory Data Analysis")
    analysis_options = ['Client Personal and Socioeconomic Attributes', 
                        'Last Contact Communication Information', 
                        'Campaign-Related Features', 
                        'Economic Context Attributes',]
    selection = st.selectbox("Select attribute to analyze", analysis_options)

    if selection:
        if selection == 'Client Personal and Socioeconomic Attributes':
            st.subheader("🧠 Client Personal and Socioeconomic Attributes")
            fig = px.histogram(df, x='age', color='y', nbins=30,
                           title='Age Distribution by Subscription Status')
            fig.update_layout(
            xaxis_title='Age',
            yaxis_title='Frequency',
            title_x=0.5
            )
            st.plotly_chart(fig)
            st.markdown("""
            <table>
            <thead>
            <tr>
            <th>Observation</th>
            <th>Business Insight</th>
            <th>Recommendation</th>
            </tr>
            </thead>
            <tbody>
            <tr>
            <td>Subscription rates tend to increase between ages 25–60 and slightly decline thereafter.</td>
            <td>Middle-aged clients are more likely to subscribe, possibly due to financial maturity and planning for retirement.</td>
            <td>Target marketing toward individuals aged 25–60 with investment-oriented messaging.</td> 
            </tr>
            </tbody>
            </table>
            """, unsafe_allow_html=True)

            fig1 = px.histogram(df, x='job', color='y',
                           title='Subscription by Job Type')
            fig1.update_xaxes(categoryorder='total descending')
            fig1.update_layout(
            xaxis_title='Job Type',
            yaxis_title='Frequency',
            title_x=0.5
            )
            st.plotly_chart(fig1)
            st.markdown("""
            <table>
            <thead>
            <tr>
            <th>Observation</th>
            <th>Business Insight</th>
            <th>Recommendation</th>
            </tr>
            </thead>
            <tbody>
            <tr>
            <td>Admin, blue-color, technician, services and management job types have higher subscription rates compared to others like entreprenuer.</td>
            <td>People with stable income or financial literacy are more investment-prone.</td>
            <td>Design job-specific campaigns (e.g., pension-related for retirees, growth for students).</td> 
            </tr>
            </tbody>
            </table>
            """, unsafe_allow_html=True)

            fig2 = px.histogram(df, x='marital', color='y',
                           title='Subscription by Marital Status')
            fig2.update_xaxes(categoryorder='total descending')
            fig2.update_layout(
            xaxis_title='Marital Status',
            yaxis_title='Frequency',
            title_x=0.5
            )
            st.plotly_chart(fig2)
            st.markdown("""
            <table>
            <thead>
            <tr>
            <th>Observation</th>
            <th>Business Insight</th>
            <th>Recommendation</th>
            </tr>
            </thead>
            <tbody>
            <tr>
            <td>Married clients show higher subscription rates.</td>
            <td>Married individuals may have more financial flexibility.</td>
            <td>Include lifestyle-specific value propositions in outreach to married couples.</td> 
            </tr>
            </tbody>
            </table>
            """, unsafe_allow_html=True)

            fig3 = px.histogram(df, x='education', color='y',
                           title='Subscription by Education Level')
            fig3.update_xaxes(categoryorder='total descending')
            fig3.update_layout(
            xaxis_title='Education Level',
            yaxis_title='Frequency',
            title_x=0.5
            )
            st.plotly_chart(fig3)

            st.markdown("""
            <table>
            <thead>
            <tr>
            <th>Observation</th>
            <th>Business Insight</th>
            <th>Recommendation</th>
            </tr>
            </thead>
            <tbody>
            <tr>
            <td>University-educated clients are more likely to subscribe.</td>
            <td>Financial literacy is likely correlated with education level.</td>
            <td>Use financial education content to nudge less-educated groups toward subscriptions.</td> 
            </tr>
            </tbody>
            </table>
            """, unsafe_allow_html=True)

            fig4 = px.histogram(df, x='default', color='y', title='Subscription by Default Status')
            fig4.update_xaxes(categoryorder='total descending')
            fig4.update_layout(
            xaxis_title='Default Status',
            yaxis_title='Frequency',
            title_x=0.5
            )
            st.plotly_chart(fig4)

            st.markdown("""
            <table>
            <thead>
            <tr>
            <th>Observation</th>
            <th>Business Insight</th>
            <th>Recommendation</th>
            </tr>
            </thead>
            <tbody>
            <tr>
            <td>Clients with existing housing loans to subscribe more.</td>
            <td>Financially constrained individuals may avoid long-term investments.</td>
            <td>Pre-qualify leads or offer flexible deposit plans to attract such clients..</td> 
            </tr>
            </tbody>
            </table>
            """, unsafe_allow_html=True)

            fig5 = px.histogram(df, x='housing', color='y', title='Subscription by Housing Loan Status')
            fig5.update_xaxes(categoryorder='total descending')
            fig5.update_layout(
            xaxis_title='Housing Loan Status',
            yaxis_title='Frequency',
            title_x=0.5
            )
            st.plotly_chart(fig5)
            st.markdown("""
            <table>
            <thead>
            <tr>
            <th>Observation</th>
            <th>Business Insight</th>
            <th>Recommendation</th>
            </tr>
            </thead>
            <tbody>
            <tr>
            <td>Clients with existing housing loans to subscribe more.</td>
            <td>Financially constrained individuals may avoid long-term investments.</td>
            <td>Pre-qualify leads or offer flexible deposit plans to attract such clients..</td> 
            </tr>
            </tbody>
            </table>
            """, unsafe_allow_html=True)

            fig6 = px.histogram(df, x='loan', color='y', title='Subscription by Personal Loan Status')
            fig6.update_xaxes(categoryorder='total descending')
            fig6.update_layout(
            xaxis_title='Personal Loan Status',
            yaxis_title='Frequency',
            title_x=0.5
            )
            st.plotly_chart(fig6)

            st.markdown("""
            <table>
            <thead>
            <tr>
            <th>Observation</th>
            <th>Business Insight</th>
            <th>Recommendation</th>
            </tr>
            </thead>
            <tbody>
            <tr>
            <td>Clients with existing personal loans to subscribe more.</td>
            <td>Financially constrained individuals may avoid long-term investments.</td>
            <td>Pre-qualify leads or offer flexible deposit plans to attract such clients..</td> 
            </tr>
            </tbody>
            </table>
            """, unsafe_allow_html=True)


        elif selection == 'Last Contact Communication Information':
            st.subheader("📞 Last Contact Communication Information")

            fig7 = px.histogram(df, x='contact', color='y', title='Subscription by Contact Method')
            fig7.update_xaxes(categoryorder='total descending')
            fig7.update_layout(
            xaxis_title='Contact Method',
            yaxis_title='Frequency',
            title_x=0.5
            )
            st.plotly_chart(fig7)

            st.markdown("""
            <table>
            <thead>
            <tr>
            <th>Observation</th>
            <th>Business Insight</th>
            <th>Recommendation</th>
            </tr>
            </thead>
            <tbody>
            <tr>
            <td>Cellular contact is more effective than telephone.</td>
            <td>Cellular outreach may offer better reach or personalization.</td>
            <td>Prioritize mobile channels for outreach campaigns.</td> 
            </tr>
            </tbody>
            </table>
            """, unsafe_allow_html=True)

            fig8 = px.histogram(df, x='month', color='y', title='Subscription by Month')
            #fig8.update_xaxes(categoryorder='total descending')
            fig8.update_layout(
            xaxis_title='Month',
            yaxis_title='Frequency',
            title_x=0.5
            )
            st.plotly_chart(fig8)
            st.markdown("""
            <table>
            <thead>
            <tr>
            <th>Observation</th>
            <th>Business Insight</th>
            <th>Recommendation</th>
            </tr>
            </thead>
            <tbody>
            <tr>
            <td>Higher subscriptions in May, through to August</td>
            <td>Campaign timing can significantly affect success.</td>
            <td>Schedule outreach based on past success patterns, especially around high-performing months like May.</td> 
            </tr>
            </tbody>
            </table>
            """, unsafe_allow_html=True)

            fig9 = px.histogram(df, x='day_of_week', color='y', title='Subscription by Day of the Week')
            #fig9.update_xaxes(categoryorder='total descending')
            fig9.update_layout(
            xaxis_title='Day of the Week',
            yaxis_title='Frequency',
            title_x=0.5
            )
            st.plotly_chart(fig9)
            st.markdown("""
            <table>
            <thead>
            <tr>
            <th>Observation</th>
            <th>Business Insight</th>
            <th>Recommendation</th>
            </tr>
            </thead>
            <tbody>
            <tr>
            <td>Higher subscriptions in May, through to August</td>
            <td>Campaign timing can significantly affect success.</td>
            <td>Schedule outreach based on past success patterns, especially on days like Monday and Thursday.</td> 
            </tr>
            </tbody>
            </table>
            """, unsafe_allow_html=True)


        elif selection == 'Campaign-Related Features':
            st.subheader("📊 Campaign-Related Features")

            fig10 = px.histogram(df, x='campaign', color='y', nbins=20,
                           title='Subscription by Number of Contacts')
            fig10.update_layout(
            xaxis_title='Number of Contacts',
            yaxis_title='Frequency',
            title_x=0.5
            )
            st.plotly_chart(fig10)

            fig11 = px.histogram(df, x='poutcome', color='y',
                           title='Subscription by Previous Outcome')
            st.plotly_chart(fig11)
            st.markdown("""
            <table>
            <thead>
            <tr>
            <th>Observation</th>
            <th>Business Insight</th>
            <th>Recommendation</th>
            </tr>
            </thead>
            <tbody>
            <tr>
            <td>Success in past campaigns does not necessary mean higher subscription.</td>
            <td>Client history is a strong indicator of future actions.</td>
            <td>Prioritize leads with positive campaign history for follow-ups.</td> 
            </tr>
            <tr>
            <td>Subscription rate declines after 4 contact attempts.</td>
            <td>Repeated contact may lead to fatigue or resistance.</td>
            <td>Limit outreach attempts and improve call quality rather than quantity.</td> 
            </tr>        
            </tbody>
            </table>
            """, unsafe_allow_html=True)
            
        elif selection == 'Economic Context Attributes':
            st.subheader("💰 Economic Context Attributes")

            fig12 = px.histogram(df, x='emp.var.rate', color='y',
                           title='Subscription by Employment Variation Rate', nbins=50)
            fig12.update_layout(
            xaxis_title='Employment Variation Rate',
            yaxis_title='Frequency',
            title_x=0.5
            )
            st.plotly_chart(fig12)

            fig13 = px.histogram(df, x='cons.price.idx', color='y',
                           title='Subscription by Consumer Price Index', nbins=50)
            fig13.update_layout(
            xaxis_title='Consumer Price Index',
            yaxis_title='Frequency',
            title_x=0.5
            )
            st.plotly_chart(fig13)

            fig14 = px.histogram(df, x='cons.conf.idx', color='y',
                           title='Subscription by Consumer Confidence Index', nbins=50)
            fig14.update_layout(
            xaxis_title='Consumer Confidence Index',
            yaxis_title='Frequency',
            title_x=0.5
            )
            st.plotly_chart(fig14)

            fig15 = px.histogram(df, x='euribor3m', color='y',
                           title='Subscription by Euribor Rate', nbins=50)
            fig15.update_layout(
            xaxis_title='Euribor Rate',
            yaxis_title='Frequency',
            title_x=0.5
            )
            st.plotly_chart(fig15)

            fig16 = px.histogram(df, x='nr.employed', color='y',
                           title='Subscription by Number of Employees', nbins=50)
            fig16.update_layout(
            xaxis_title='Number of Employees',
            yaxis_title='Frequency',
            title_x=0.5
            )
            st.plotly_chart(fig16)


            st.markdown("### Observation and Business Insight:")

            st.markdown("""
            <table>
            <thead>
            <tr>
            <th>Feature</th>
            <th>Observation</th>
            <th>Business Insight</th>
            <th>Recommendation</th>
            </tr>
            </thead>
            <tbody>
            <tr>
            <td><code>emp.var.rate</code></td>
            <td>Positive employment variation leads to more subscriptions.</td>
            <td>People invest during uncertain job markets.</td>
            <td>Promote term deposits as safe-haven during downturns.</td>
            </tr>
            <tr>
            <td><code>cons.price.idx</code></td>
            <td>Higher CPI slightly correlates with fewer subscriptions.</td>
            <td>Inflation may reduce investment interest.</td>
            <td>Offer inflation-protected deposit options.</td>
            </tr>
            <tr>
            <td><code>cons.conf.idx</code></td>
            <td>More negative confidence → more subscriptions.</td>
            <td>People seek stability in uncertain times.</td>
            <td>Emphasize stability and guaranteed returns.</td>
            </tr>
            <tr>
            <td><code>euribor3m</code></td>
            <td>Higher rates → higher subscriptions.</td>
            <td>Lower bank interest rates encourage locking funds in term deposits.</td>
            <td>Market term deposits more when interest rates drop.</td>
            </tr>
            <tr>
            <td><code>nr.employed</code></td>
            <td>Higher employment → more subscriptions.</td>
            <td>Again suggests preference for security in hard times.</td>
            <td>Position product as a safety net during job uncertainty.</td>
            </tr>
            </tbody>
            </table>
            """, unsafe_allow_html=True)

# Train Model Page
elif selected == "Train Model":
    # Sidebar instructions
    st.sidebar.markdown("""
    **Train Model Page Instructions**  
    - Select relevant features using checkboxes.  
    - Choose a classification model.  
    - Click "Train" to evaluate model performance.  
    - Review confusion matrix, scores, and download the model.
    """)


    st.title("Train a Classification Model")

    # Feature selection
    st.subheader("1. Feature Selection")
    feature_options = [col for col in df.columns if col != 'y']
    selected_features = st.multiselect("Select features to include in training", feature_options)

    # Model selection
    st.subheader("2. Model Selection")
    model_options = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Naive Bayes": GaussianNB(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "LightGBM": LGBMClassifier(random_state=42),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
        #"Support Vector Machine": SVC(probability=True)
    }
    selected_model_name = st.selectbox("Choose a classifier", list(model_options.keys()))

    # Train model
    if st.button("Train Model"):
        with st.spinner("Training your model..."):
            model = model_options[selected_model_name]
            X = pd.get_dummies(df[selected_features])
            y = df['y'].apply(lambda x: 1 if x == 'yes' else 0)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.success("Your model is ready")

            # Metrics
            report = classification_report(y_test, y_pred, output_dict=True)
            st.subheader("Model Performance")
            st.write(f"Accuracy: {report['accuracy']:.2f}")
            st.write(f"Precision: {report['1']['precision']:.2f}")
            st.write(f"Recall: {report['1']['recall']:.2f}")
            st.write(f"F1-Score: {report['1']['f1-score']:.2f}")

            # Confusion matrix
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
            st.pyplot(fig)

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                st.subheader("Top 10 Feature Importances")
                importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
                fig = plt.figure()
                sns.barplot(x=importances[:10], y=importances.index[:10])
                plt.xlabel("Importance")
                plt.ylabel("Features")
                st.pyplot(fig)

            # Save and download model
            buffer = BytesIO()
            pickle.dump(model, buffer)
            st.download_button("Download Model", buffer.getvalue(), file_name="trained_model.pkl")

# Make Prediction Page
elif selected == "Make Prediction":
    # Sidebar instructions
    st.sidebar.markdown("""
    **Prediction Page Instructions**  
    - Choose input values using dropdowns and sliders.  
    - Click "Predict" to get subscription outcome.  
    - View the prediction with performance confidence.
    """)
    
    # Load trained model and metadata
    with open("model.pkl", "rb") as file:
        model, feature_columns, metrics = pickle.load(file)
    
    st.title("🔮 Make Prediction")

    st.markdown("Provide the client's details to predict the likelihood of subscription to a term deposit.")

    # Define inputs
    age = st.slider("Pick Age", min_value=18, max_value=95, step=1)
    campaign = st.slider("Pick Number of Contacts During Campaign", 1, 60, step=1)
    previous = st.slider("Pick Number of Contacts Before Campaign", 0, 10, step=1)
    pdays = st.slider("Pick Days Since Last Contact", -1, 999, step=1)
    euribor3m = st.slider("Pick Euribor 3-Month Rate", 0.5, 5.0, step=0.1)
    nr_employed = st.slider("Pick Number of Employees", 4900, 5300, step=10)

    job = st.selectbox("Select Job", df['job'].unique())

    marital = st.selectbox("Select Marital Status", df['marital'].unique())
    education = st.selectbox("Select Education Level", df['education'].unique())
    contact = st.selectbox("Select Contact Communication Type", df['contact'].unique())
    default = st.selectbox("Select Default Status", df['default'].unique())
    month = st.selectbox("Select Contact Month", df['month'].unique())
    housing = st.selectbox("Select Housing Loan Status", df['housing'].unique())
    day_of_week = st.selectbox("Select Day of Week", df['day_of_week'].unique())
    loan = st.selectbox("Select Personal Loan Status", df['loan'].unique())
    poutcome = st.selectbox("Select Previous Outcome", df['poutcome'].unique())
    
    # Feature Engineering
    df['contacted_before'] = df["pdays"].apply(lambda x: 0 if x == 999 else 1)
    contacted_before = df['contacted_before'].values[0]

    # Create input dataframe
    input_dict = {
        'age': age,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'euribor3m': euribor3m,
        'nr.employed': nr_employed,
        'contacted_before': contacted_before,
        'job_' + job: 1,
        'marital_' + marital: 1,
        'education_' + education: 1,
        'default_' + default: 1,
        'housing_' + housing: 1,
        'loan_' + loan: 1,
        'contact_' + contact: 1,
        'month_' + month: 1,
        'day_of_week_' + day_of_week: 1,
        'poutcome_' + poutcome: 1,
    
    }
    # Ensure all feature columns are present
    input_df = pd.DataFrame([input_dict])
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_columns]

    if st.button("Make Prediction"):
        with st.spinner("Training your model..."):

            prediction = model.predict(input_df)[0]
            label = "likely to subscribe" if prediction == 1 else "not likely to subscribe"
            
            st.success(f"📊 The client is **{label}** with:")
            st.markdown(f"""
            - **Accuracy:** {metrics['accuracy']:.2f}  
            - **Precision:** {metrics['precision']:.2f}  
            - **Recall:** {metrics['recall']:.2f}  
            - **F1 Score:** {metrics['f1']:.2f}
            """)

# About Page
elif selected == "About":
    st.title("About This App")
    st.markdown("""
    **Term Deposit Subscription Predictor** was built to help marketing teams identify clients more likely to subscribe to term deposits.
    It uses machine learning to analyze client data and predict subscription likelihood, enabling targeted marketing strategies.
    The app provides an interactive interface for data exploration, model training, and making predictions.
    
    **This is part of my delivarables for the Azubi Africa Talent Mobility Program (TMP) Technical Fit Assessment.**
    
    **Project Repository:** [GitHub](https://github.com/SirGamah/Azubi-Africa-TMP-Technical-Fit-Assessment/tree/main)
    
    **Developer:** Teye Richard Gamah  
    **Contact:** [LinkedIn](https://linkedin.com/in/gamah/) | [Email](mailto:gamahrichard5@gmail.com)
    """)
