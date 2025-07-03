# Rainfall Prediction Project

This project is part of **Module 13** of the Data Science curriculum and focuses on predicting **precipitation levels** using weather attributes from the **Austin Weather Dataset**.

##  Objective

To build a **Linear Regression** model that predicts the amount of rainfall (`PrecipitationSumInches`) based on features such as:
- Temperature
- Humidity
- Dew Point
- Visibility
- Wind Speed

Additionally, the project visualizes trends and correlations in the weather data to understand how various factors influence rainfall.

---

##  Dataset

- **File Name**: `austin_weather.csv`
- **Source**: LMS > Module 12 > Project - COVID-19 Analysis > Case Study - Part 1 > Resource 1

---

##  Technologies Used

- **Python**
- **Jupyter Notebook**
- **Pandas** for data handling
- **NumPy** for numerical operations
- **Matplotlib** and **Seaborn** for data visualization
- **Scikit-learn** for Linear Regression modeling

---

##  Project Steps

### 1. Importing Libraries
Load all required libraries such as pandas, seaborn, matplotlib, and scikit-learn.

### 2. Loading the Dataset
Read the CSV file `austin_weather.csv` into a Pandas DataFrame.

### 3. Data Cleaning
- Replaced `"T"` (trace amounts of rain) with `0.0`
- Replaced `"-"` with NaN
- Converted columns to appropriate data types
- Dropped irrelevant columns like `Events` and `Date`
- Removed rows with missing values

### 4. Correlation Analysis
Plotted a heatmap to find relationships between features.

### 5. Model Building
- Selected features (`X`) and target (`y`)
- Split data into training and testing sets
- Applied **Linear Regression**

### 6. Evaluation
Used:
- **Mean Squared Error**
- **R2 Score**  
to evaluate model performance.

### 7. Visualization
- Plotted **Actual vs Predicted** rainfall
- (Optional) Displayed **Precipitation over Time** using `Date`

---

##  Results

- A regression model that predicts rainfall based on weather parameters
- Insightful visualizations showing trends and relationships

---

## How to Run

1. Clone or download the project folder.
2. Make sure `austin_weather.csv` is in the same directory as your `.ipynb` notebook.
3. Open `Rainfall_Prediction.ipynb` in **Jupyter Notebook** or **JupyterLab**.
4. Run the notebook cells step by step.


