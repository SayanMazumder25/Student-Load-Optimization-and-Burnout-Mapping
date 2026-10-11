# Student Load Optimization & Burnout Mapping

### A Data Science Project on Student Workload, Academic Performance, and Stress

**Author:** Sayan Mazumder
**Project Type:** Exploratory Data Analysis & Machine Learning
**Status:** Completed — Initial Analysis and Predictive Modelling

---

## 📌 Project Overview

Balancing academic responsibilities, extracurricular activities, and physical fitness can be challenging for students. Excessive workload may compete with sleep and recovery, potentially affecting stress levels and academic performance.

**Student Load Optimization & Burnout Mapping** is a data science project that explores the relationships between students' daily activities, recovery patterns, reported stress, and academic performance.

The project uses data cleaning, exploratory data analysis (EDA), statistical visualisation, correlation analysis, and a Random Forest regression model to investigate patterns in student workload and reported stress.

The objective is to understand how different lifestyle and workload variables relate to one another and explore whether machine learning can predict reported stress levels from selected features.

> **Important:** This project uses synthetic data. Its findings illustrate patterns within the generated dataset and should not be interpreted as established conclusions about real students or clinical burnout.

---

## 🎯 Objectives

The project focuses on five primary objectives:

1. **Data Cleaning:** Handle duplicate records, missing values, and inconsistent numerical data types.
2. **Exploratory Data Analysis:** Visualise relationships between workload, sleep, stress, and academic performance.
3. **Correlation Analysis:** Examine associations among numerical variables.
4. **Workload Pattern Analysis:** Explore how combined physical training and extracurricular activity hours relate to academic scores.
5. **Predictive Modelling:** Train and evaluate a Random Forest regression model to estimate reported stress levels.

---

## 📊 Dataset Description

The project uses a synthetic dataset representing student activities, academic outcomes, and reported stress.

The primary dataset contains the following variables:

| Variable                       | Description                                                                 |
| ------------------------------ | --------------------------------------------------------------------------- |
| `Daily_Study_Hours`            | Daily time devoted to studying                                              |
| `Gym_Hours`                    | Daily time devoted to physical training (e.g., gym workouts)                |
| `Debate_Prep_Hours`            | Daily time devoted to extracurricular activities (e.g., debate preparation) |
| `Average_Sleep_Hours`          | Average duration of sleep                                                   |
| `Academic_Score`               | Academic performance score                                                  |
| `Reported_Stress_Level`        | Reported stress rating                                                      |
| Additional numerical variables | Other numerical fields present in the dataset                               |

The notebook also explores supplementary datasets related to study hours, extracurricular activities, physical training, sleep, and stress.

**Data source:** Synthetic data generated for educational and analytical purposes.

The dataset is not a representative survey of real students.

---

## 🧹 Data Preprocessing

The data-cleaning workflow includes:

* Removing duplicate records.
* Converting relevant columns to numerical data types.
* Coercing invalid numerical entries to missing values.
* Imputing missing values using column means.
* Filtering academic scores to retain values at or below 100.
* Rounding the cleaned dataset to one decimal place.
* Exporting the processed dataset for subsequent analysis.

These operations prepare the data for exploratory analysis and modelling.

Mean imputation is a simplifying choice for this project. Alternative approaches may be more appropriate for discrete ratings or for a predictive pipeline intended for real-world use.

---

## 📈 Exploratory Data Analysis

The notebook includes several visual analyses.

### 1. Study Hours vs. Stress

A two-dimensional histogram examines the distribution of observations across daily study hours and mean stress levels.

The colour intensity represents the number of observations within each bin.

### 2. Extracurricular Activity Hours vs. Stress

A two-dimensional histogram explores how extracurricular activity hours are distributed in relation to reported stress, using debate preparation as an example.

### 3. Physical Training vs. Stress

A two-dimensional histogram examines the relationship between gym hours and stress levels.

### 4. Sleep vs. Stress

A two-dimensional histogram explores the distribution of sleep duration and reported stress.

### 5. Correlation Heatmap

A correlation heatmap visualises linear relationships between numerical variables, helping identify positive and negative associations.

Correlation alone does not establish causation.

### 6. Combined Workload vs. Academic Performance

The project defines combined extracurricular workload as:

**Total Load = Gym Hours + Debate Preparation Hours**

A line plot explores how academic scores vary with this combined workload.

### 7. Workload Allocation Matrix

A scatter plot places debate preparation hours against gym hours, with colour representing academic scores.

This visualisation helps explore how different allocations of time appear within the synthetic dataset.

### 8. Sleep, Academic Performance, and Stress

A scatter plot examines the relationship between average sleep duration and academic scores, using colour to represent reported stress levels.

---

## 🤖 Machine Learning

### Model: Random Forest Regressor

The project uses a Random Forest regression model from Scikit-learn to predict reported stress levels.

Random Forest is an ensemble learning method that combines predictions from multiple decision trees.

### Features

The notebook selects five input features from the cleaned dataset:

* Daily study hours
* Gym hours
* Debate preparation hours
* Average sleep hours
* Academic score

### Target Variable

`Reported_Stress_Level`

### Training Configuration

The notebook uses:

* **Algorithm:** Random Forest Regressor
* **Number of trees:** 100
* **Random state:** 42
* **Test set proportion:** 20%

The model is evaluated using a held-out test set.

---

## 📏 Model Evaluation

The current recorded evaluation results are:

| Metric                             |  Result |
| ---------------------------------- | ------: |
| R² Score                           |  0.3961 |
| Mean Absolute Error (MAE)          | 0.83821 |
| Predictions within ±1 stress point |  64.25% |

### Interpretation

**R² Score — 0.3961**

The model explains approximately 39.61% of the variation in the target values on the evaluated test set.

**Mean Absolute Error — 0.83821**

The predictions differ from the actual stress ratings by an average absolute error of approximately 0.84 points.

**Predictions within ±1 stress point — 64.25%**

Approximately 64.25% of predictions fall within one point of their corresponding actual target values, according to the notebook's evaluation metric.

The last metric is a tolerance-based agreement measure, not conventional classification accuracy.

These results describe the current recorded evaluation. They should be treated as final results only after confirming that the notebook has been executed sequentially and that the reported dataset sizes and metrics correspond to the same run.

Because the dataset is synthetic, these metrics do not establish predictive performance on real students.

---

## 📉 Model Diagnostics

The notebook includes two diagnostic visualisations.

### Actual vs. Predicted Values

This scatter plot compares actual stress ratings with model predictions. A reference line indicates where predictions would equal the actual values.

### Residual Analysis

Residuals are calculated as:

**Residual = Actual Value − Predicted Value**

The residual plot helps examine prediction errors and identify possible patterns that may indicate limitations in the model.

The saved model is exported as `burnout_model.joblib`.

---

## 🛠️ Technology Stack

| Technology       | Purpose                                |
| ---------------- | -------------------------------------- |
| Python           | Programming and analysis               |
| Pandas           | Data manipulation and cleaning         |
| NumPy            | Numerical operations                   |
| Matplotlib       | Data visualisation                     |
| Seaborn          | Statistical visualisation              |
| Scikit-learn     | Machine learning and evaluation        |
| Joblib           | Model serialisation                    |
| Jupyter Notebook | Interactive analysis and documentation |

---

## 🚀 Getting Started

### Prerequisites

Install Python and the libraries used in the notebook.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib jupyter
```

### Clone the Repository

```bash
git clone https://github.com/SayanMazumder25/Student-Load-Optimization-and-Burnout-Mapping.git
```

Navigate to the project directory:

```bash
cd Student-Load-Optimization-and-Burnout-Mapping
```

### Run the Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open `Project-1.ipynb` and execute the cells sequentially.

**Note:** The notebook references several CSV datasets. Ensure that all required files are present in the expected directory before executing the analysis.

The notebook also contains a reference to `test1.xlsx`. Check that this file is available if the corresponding cell is retained.

---

## 🔍 Limitations

This project has several limitations:

1. **Synthetic data:** The generated observations do not establish that the same patterns occur in real student populations.
2. **Exploratory relationships:** Correlations and visual patterns do not establish causal relationships.
3. **Model performance:** The current R² indicates that a substantial portion of the target's variation remains unexplained.
4. **Feature and data limitations:** Results depend on the variables, assumptions, and data-generation process used.
5. **Workload thresholds:** Any apparent threshold or preferred allocation is specific to the dataset and analysis; it is not a validated universal limit.
6. **Evaluation scope:** Performance on synthetic test data does not guarantee performance on real-world data.

The project should be viewed as an educational demonstration of data analysis and predictive modelling, not as a validated burnout detection or diagnostic system.

---

## 🔮 Future Scope

Potential extensions include:

* Evaluating the model against a simple baseline predictor.
* Comparing Random Forest with other regression algorithms.
* Investigating feature importance and model interpretability.
* Improving the preprocessing and model-training pipeline.
* Exploring alternative methods for identifying workload patterns.
* Validating the analysis on appropriately collected real-world data.
* Developing an interactive dashboard for exploring workload and recovery patterns.

Future improvements should be guided by the quality of the data and the specific analytical question rather than by model scores alone.

---

## 👨‍💻 Author

**Sayan Mazumder**

B.Tech in Information Technology
Government College of Engineering & Textile Technology, Serampore

B.S. in Data Science and Applications
Indian Institute of Technology, Madras

Interested in Data Science, Machine Learning, Exploratory Data Analysis, and applied analytical problem-solving.

---

## ⚠️ Disclaimer

This project is intended for educational and exploratory purposes only.

It is not a clinical assessment tool and should not be used to diagnose burnout, determine an individual's mental health status, or prescribe health-related interventions.

All interpretations must be considered in the context of the synthetic dataset used in the analysis.