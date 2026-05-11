# Cafe Sales Data Cleaning and Revenue Prediction

This project explores a real-world cafe sales dataset and builds a simple machine learning model to estimate transaction revenue.

## Project Goal

The goal of this project was to:

* clean messy transaction data
* handle missing values and invalid entries
* prepare the dataset for analysis
* build a basic machine learning model to predict `Total Spent`

## Dataset

The dataset contains **10,000 transactions** and includes the following columns:

* Transaction ID
* Item
* Quantity
* Price Per Unit
* Total Spent
* Payment Method
* Location
* Transaction Date

The raw dataset contained missing values, invalid values such as `ERROR` and `UNKNOWN`, and missing dates.

## Data Cleaning

The cleaning process included:

* converting numeric columns to proper numeric types
* replacing invalid values with `NaN`
* filling numeric missing values with the median
* filling categorical missing values with the mode
* converting `Transaction Date` to datetime format
* creating a new `month` feature
* removing rows with missing transaction dates

## Machine Learning Model

A **Linear Regression** model was built using:

**Features**

* Quantity
* Price Per Unit
* month

**Target**

* Total Spent

The dataset was split into:

* 80% training data
* 20% testing data

## Results

The model produced the following results:

* **Mean Absolute Error (MAE):** 1.67
* Example prediction: for `Quantity = 3`, `Price Per Unit = 4`, `month = 7`, predicted revenue was **11.79**

## What I Learned

Through this project I practiced:

* real-world data cleaning
* feature selection
* train/test splitting
* linear regression
* prediction
* model evaluation using MAE
* interpreting model coefficients

## Notes

One important observation was that `month` had very little impact on the prediction, while `Quantity` and `Price Per Unit` were much stronger predictors.

This was my early machine learning practice project, focused on understanding the complete workflow from raw data to model evaluation.
