''''
Author: Hridoy

This is a loan app where different ML model and Exploratory data analysis has been performed  using freely available dataset



'''
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import streamlit as st
import pandas as pd
# from lazypredict.Supervised import LazyRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.datasets import load_diabetes, load_boston
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
#---------------------------------#
# Page layout

from PIL import Image
import streamlit as st

im = Image.open("loan.png")
st.set_page_config(
    page_title="The Machine Learning Algorithm App 🌱",
    page_icon=im,
    layout="wide",
)
## Page expands to full width
# st.set_page_config(page_title='The Machine Learning Algorithm App 🌱',
#     layout='wide')
#---------------------------------#
# Model building
def build_model(df):
    # df = df.loc[:100] # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
    X = df.iloc[:,1:-1] # Using all column except for the last column as X
    Y = df.iloc[:,-1] # Selecting the last column as Y
    # X=X.drop(columns=['Dependents'])
    st.markdown('**1.2. Dataset dimension**')
    st.write('X')
    st.info(X.shape)
    st.write('Y')
    st.info(Y.shape)

    st.markdown('**1.3. Variable details**:')
    st.write('X variable (first 20 are shown)')
    st.info(list(X.columns[:20]))
    st.write('Y variable')
    st.info(Y.name)
    #
    categorical_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
    #Impute missing values with mode
    X['Dependents'].replace('3+', 4, inplace=True)
    X['Dependents'] = X['Dependents'].astype(float)
    X['Self_Employed'].fillna('Unknown', inplace=True)
    # X['Self_Employed'].fillna(X['Self_Employed'].mode(), inplace=True)
    X['Gender'].fillna(X['Gender'].mode()[0], inplace=True)
    X['Dependents'].fillna(X['Dependents'].mean(), inplace=True)
    X['Credit_History'].fillna(X['Credit_History'].mode()[0], inplace=True)
    X['CoapplicantIncome'].fillna(X['CoapplicantIncome'].mean(), inplace=True)
    X['Loan_Amount_Term'].fillna(X['Loan_Amount_Term'].median(), inplace=True)
    null_columns = X.columns[X.isnull().any()]
    st.info(null_columns)
###

    # Convert categorical columns to one-hot encoding
    X = pd.get_dummies(X, columns=categorical_columns)


    # Build lazy model
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Define a list of classifiers
    classifiers = [
        ('Logistic Regression', LogisticRegression()),
        ('Decision Tree', DecisionTreeClassifier()),
        ('Random Forest', RandomForestClassifier()),
        ('Support Vector Classifier', SVC()),
        ('K-Nearest Neighbors', KNeighborsClassifier())
    ]

    # Initialize lists to store accuracy scores
    train_accuracy = []
    test_accuracy = []

    # Iterate over classifiers
    for name, clf in classifiers:
        # Train the model
        clf.fit(X_train, y_train)

        # Predictions on training and test data
        train_preds = clf.predict(X_train)
        test_preds = clf.predict(X_test)

        # Calculate accuracy scores
        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)

        # Append accuracy scores to lists
        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)

    # Plotting using Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    index = range(len(classifiers))

    bar1 = ax.bar(index, train_accuracy, bar_width, label='Train Accuracy')
    bar2 = ax.bar([i + bar_width for i in index], test_accuracy, bar_width, label='Test Accuracy')

    ax.set_xlabel('Classifier')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy of Different Classifiers on Training and Test Data')
    ax.set_xticks([i + bar_width / 2 for i in index])
    ax.set_xticklabels([name for name, _ in classifiers])
    ax.legend()
    st.pyplot(fig)

# Download CSV data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

def imagedownload(plt, filename):
    s = io.BytesIO()
    plt.savefig(s, format='pdf', bbox_inches='tight')
    plt.close()
    b64 = base64.b64encode(s.getvalue()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

#---------------------------------#
st.write("""
# The Machine Learning Algorithm Comparison App 🌱


""")

#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
""")

# Sidebar - Specify parameter settings
with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
    # seed_number = st.sidebar.slider('Set the random seed number', 1, 100, 42, 1)


#---------------------------------#
# Main panel

# Displays the dataset
st.subheader('1. Dataset')



if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df)






    # df = load_csv()
    pr = ProfileReport(df, explorative=True)
    st.header('**Input DataFrame**')
    st.write(df)
    st.write('---')
    st.header('**Pandas Profiling Report**')
    st_profile_report(pr)
    build_model(df)
# else:
#     st.info('Awaiting for CSV file to be uploaded.')
#     if st.button('Press to use Example Dataset'):
#         # Diabetes dataset
#         #diabetes = load_diabetes()
#         #X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
#         #Y = pd.Series(diabetes.target, name='response')
#         #df = pd.concat( [X,Y], axis=1 )
#
#         #st.markdown('The Diabetes dataset is used as the example.')
#         #st.write(df.head(5))
#
#         # Boston housing dataset
#         boston = load_boston()
#         #X = pd.DataFrame(boston.data, columns=boston.feature_names)
#         #Y = pd.Series(boston.target, name='response')
#         X = pd.DataFrame(boston.data, columns=boston.feature_names).loc[:100] # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
#         Y = pd.Series(boston.target, name='response').loc[:100] # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
#         df = pd.concat( [X,Y], axis=1 )
#
#         st.markdown('The Boston housing dataset is used as the example.')
#         st.write(df.head(5))
#
#         build_model(df)
