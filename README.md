---
title: "Assigment - KNN DIY"
author:
- Le Minh Quan - Author
- Maximilian - Reviewer
date: "`r format(Sys.time(), '%d %B, %Y')`"
output:
  html_notebook:
  toc: true
  toc_depth: 2
---

```{r}
library(readr)
library(tidyverse)
library(tm)
library(caret)
library(wordcloud)
library(e1071)
library(creditmodel)
```

## Business Understanding

This KNN model is launched to detect the possibility of getting diabetes among a specific amount of people to preventing it

## Data Understanding

Accessing to the data

```{r}
rawDF <- read_csv("datasets/KNN-diabetes.csv")
```

According to the data set, we have the following factors:
  - Pregnancies: This refers to the number of times a patient has been pregnant.
  - Glucose: This is the patient's plasma glucose concentration in a 2-hour oral glucose tolerance test.
  - BloodPressure: This refers to the patient's diastolic blood pressure in millimeters of mercury (mm Hg).
  - SkinThickness: This is the thickness of the patient's skin in millimeters at the triceps site.
  - Insulin: This is the patient's 2-hour serum insulin level measured in microunits per milliliter (mu U/ml).
  - BMI: This is the patient's body mass index, which is calculated by dividing their weight in kilograms by their height in meters squared.
  - DiabetesPedigreeFunction: This is a function that measures the likelihood of diabetes based on family history.
  - Age: This is the patient's age in years.
  - Outcome: This refers to whether the patient has diabetes or not, with a value of 0 indicating the patient does not have diabetes and a value of 1 indicating the patient does have diabetes.

## Data Preparation

The Data set consist of all of the important factor, thereby we won't deleting anything (but will select them later)

```{r}
cleanDF <- rawDF 
head(cleanDF)
```

We counting the outcomes to see how many patients with Diabetes or with No Diabetes and the percentage

```{r}
cntOutc <- table(cleanDF$Outcome) 
propOutc <- round(prop.table(cntOutc) * 100 , digits = 1)
cntOutc 
propOutc
```

The "Outcome" is the variable that we need to predict

```{r}
cleanDF$Outcome <- factor(cleanDF$Outcome, levels = c("0", "1"), labels = c("No Diabetes", "Diabetes")) %>% relevel("Diabetes") 
head(cleanDF,10)
```

At here, I selected these 4 variables because they are more correlated to the "Outcome" than the others

```{r}
summary(cleanDF[c("Glucose", "Pregnancies", "BMI", "BloodPressure",
"DiabetesPedigreeFunction")])
```

We see the variables have very different range thereby we will apply normalization to rescale all features to a standard range of values.

```{r}
cleanDF_n <- sapply(1:8, function(x) { normalize(cleanDF[,x]) }) %>% as.data.frame() 
colnames(cleanDF_n) <- colnames(cleanDF)[1:8]
summary(cleanDF_n[c("Glucose", "Pregnancies", "BMI", "BloodPressure",
"DiabetesPedigreeFunction")])
```

We split the data into 2 parts: Training sets and Test sets (by using "train_test_split" from package "creditmodel")

```{r}
test <- train_test_split(cleanDF)[[1]]
train <- train_test_split(cleanDF)[[2]]
test_feat <- test[-9]
train_feat <- train[-9]
test_labels <- test[, 9]
train_labels <- train[,9]
```

## Modeling

We just need one function from the class package to train the KNN model. The function returns a collection of predictions after applying the trained model to the set of test features.

```{r}
library(class)
cleanDF_test_pred <- knn(train = as.matrix(train_feat), test = as.matrix(test_feat), cl = as.matrix(train_labels), k=10)
head(cleanDF_test_pred)
```

Here is our own table:

```{r}
confusionMatrix(cleanDF_test_pred, test_labels[[1]], positive = NULL,
dnn = c("True", "Prediction"))
```

## Evaluation and Deployment

Overall: 
  - The accuracy percentage is 75.65%. 
  - 19.13% of test feature has True Positive and 56.52% is True Negative. 
  - 3.04% is False Positive and 21.3% is False Negative.

Of the 230 cases, 93 are positive (44 TPs and 49 FNs) and 137 are negative (7 FPs and 130 TNs).

Of 137 negative diabetes, 94.89% are correctly predicted. This is a good result. 

However, of 93 positive cases, only 47.31% are correctly identified. More than 50% of the cases go wrong which is a bad outcome.

Precision in this model is 86.27% which tells us that there is a 86.27% correct if it predicts someone is positive, and the Recall in this model is 47.31%. 

Therefore, the Precision is good but it still have room to improve and the Recall is quite high
which is not so good.
