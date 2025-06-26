# PREDICTIVE-ANALYSIS-USING-MACHINE-LEARNING

*COMPANY* : CODTECH IT SOLUTIONS

*NAME*:BANOTH KAVYA

*INTERN ID*:CT06DK441

*DOMAIN*:DATA ANALYTICS

*DURATION*: 6 WEEKS

*MENTOR*:NEELA SANTHOSH

**📊 Startup Profit Prediction using Linear Regression**
This project performs profit prediction for startups using a multiple linear regression model. The model is trained on a dataset of 50 startups with features such as R&D Spend, Administration, Marketing Spend, and State. The objective is to predict the profit based on investment decisions.

**🔍 Objective**
To build a supervised machine learning model that can:

Analyze how different types of spending affect startup profits.

Predict profit using multivariable linear regression.

Evaluate model performance using MSE and R² metrics.

Visualize insights and prediction accuracy.

**📁 Dataset Information**
**Dataset**: 50_Startups.csv
Source: Common educational dataset for regression problems

**Features:**

R&D Spend: Capital spent on research and development

Administration: Administrative costs

Marketing Spend: Expenditure on marketing

State: Categorical location feature (e.g., California, New York)

Profit: Target variable (label) to be predicted

**⚙️ Technologies Used**
Python

Pandas & NumPy – Data manipulation

Matplotlib & Seaborn – Data visualization

Scikit-learn – Model building and evaluation

**🧪 Project Workflow**
**✅ 1. Importing Libraries**
Essential Python libraries are imported for data loading, preprocessing, visualization, and machine learning.

**✅ 2. Data Loading**
The dataset is read from a CSV file using pandas.read_csv() and basic information is printed:

Shape of the dataset

First 5 rows

Data types

Summary statistics

**✅ 3. Data Preprocessing**
Checked for missing values.

Encoded the State categorical column using Label Encoding to convert strings to numerical values.

**✅ 4. Exploratory Data Analysis (EDA)**
A correlation heatmap was created using Seaborn to show relationships between numerical variables. This helps identify which spending category correlates most with profit.

**✅ 5. Model Building**
Split the data into training (80%) and testing (20%) sets using train_test_split().

Trained a Linear Regression model using LinearRegression() from scikit-learn.

**✅ 6. Prediction & Evaluation**
Predicted profits using the model on test data.

Calculated:

Mean Squared Error (MSE)

R² Score (coefficient of determination)

**✅ 7. Visualization**
Generated a scatter plot comparing actual vs predicted profits with a diagonal red dashed line for reference. This visualization helps evaluate the closeness of predictions.

**📈 Sample Output**
yaml
Copy
Edit
Model Performance:
Mean Squared Error: 7981252.34
R² Score: 0.92
This indicates that the model explains 92% of the variance in profit prediction, which is a strong performance for a linear model.

**📂 Output Files**
task2_corr_startup.jpg: Correlation heatmap

task2_startup.jpg: Actual vs Predicted profit scatter plot

**▶️ How to Run the Project**
Ensure you have Python installed.

Install the required packages:

bash
Copy
Edit
pip install pandas numpy matplotlib seaborn scikit-learn
Make sure the 50_Startups.csv file is located in your Downloads/ folder or adjust the path in the code.

Run the script using:

bash
Copy
Edit
python startup_profit_predict.py
**🔬 Future Enhancements**
Use OneHotEncoding for better handling of categorical data (State).

Apply Regularized Regression (Lasso, Ridge) to reduce overfitting.

Develop a web interface using Streamlit or Flask for interactive predictions.

Save and load the trained model using joblib.

**📁 Folder Structure**
Copy
Edit
StartupProfitPrediction/
├── 50_Startups.csv
├── startup_profit_predict.py
├── task2_corr_startup.jpg
├── task2_startup.jpg
└── README.md
**👩‍💻 Author**
Kavya Banoth
This project was completed as part of an internship to demonstrate knowledge of data preprocessing, regression modeling, and visualization in Python.
