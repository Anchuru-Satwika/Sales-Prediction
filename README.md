<h1 align="center" font-size:16px"><b>Sales-Prediction</b></h1>


## Aim

This project aims to predict sales based on advertising expenditure using the given dataset. The dataset contains information about advertising spending on different platforms (TV, Radio, and Newspapers) and the corresponding sales amount.

## Libraries Used

The following important libraries were used for this project:

- NumPy
- Pandas
- matplotlib.pyplot
- seaborn
- sklearn.model_selection.train_test_split
- sklearn.linear_model.LinearRegression

## Dataset

The dataset was loaded using pandas as a DataFrame from the file "advertising.csv".

## Data Exploration and Preprocessing

1. The shape and descriptive statistics for the dataset were displayed using `df.shape` and `df.describe()`.
2. A pair plot was created to visualize the relationship between advertising expenditure on TV, Radio, Newspaper, and sales using `seaborn.pairplot`.
3. Histograms were plotted to observe the distribution of advertising expenditure on TV, Radio, and Newspaper using `matplotlib.pyplot.hist`.

## Correlation Analysis

1. A correlation matrix heatmap was plotted to observe the correlation between advertising expenditure on TV, Radio, Newspaper, and sales using `seaborn.heatmap`.

## Model Training

1. The data was split into training and testing sets using `train_test_split`.
2. Linear regression model was trained on the training data using `sklearn.linear_model.LinearRegression`.

## Model Prediction

1. The model was used to predict sales based on advertising expenditure on TV for the test set using `model.predict(X_test)` and the corresponding advertising expenditure on TV in the test set.
2. The model coefficients and intercept were obtained using `model.coef_` and `model.intercept_`.
3. The predictions and actual sales values were plotted using `matplotlib.pyplot.plot` and `matplotlib.pyplot.scatter`.

**Competition Website**: [kaggle](https://www.kaggle.com/code/ashydv/sales-prediction-simple-linear-regression/input)
