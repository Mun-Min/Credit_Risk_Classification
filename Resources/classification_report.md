# Credit Risk Analysis Report

<p align="center">

  <img src="../Images/credit_risk.jpg" alt="credit risk image" height = 400 width = 800>

</p>

</br>

## Table of Contents: 

1. [Overview of the Analysis](#overview-of-the-analysis)

2. [Model Results](#results)

3. [Summary](#summary)

</br>

## **Overview of the Analysis**

Lending companies lend money/properties to borrowers with the expectation that the borrower will either return the asset or repay the lender. Credit Risk is associated with a borrower not returning an asset or paying a loan back causing a lender to lose money. This is measured by lenders in many ways, however in this analysis we will use Machine Learning to analyze a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers. 

</br>

* Using a machine learning model, I will try to determine which loans are healthy (low-risk) or non-healthy (high-risk) based on the loan status provided by the lending company. 

  * The Logistic Regression Algorithm is the best tool to use for our machine learning model since it is widely used to predict the probability of a target variable in classification problems.  

</br>

* Using the dataset provided by the lending company, I created a Logistic Regression Model that generated an accuracy score of 95%. Although the model generated a high-accuracy, the models recall value (0.91) for non-healthy loans is lower than the recall value (0.99) for healthy loans. This indicates that the model will predict loan status's as healthy better than being able to predict loan status's as non-healthy. This is due to the dataset being imbalanced, meaning that most of the data belongs to one class label (in this case healthy loans greatly outweighed non-healthy loans).

</br>

`Taking a look at the code in step 3 [Split the Data into Training and Testing Sets], using the value_counts function, we are able to see that the data is highly imbalanced. The majority class is healthy loans [0] and the minority class is non-healthy loans [1]:`

```
# code
y.value_counts()

# output
0    75036
1     2500
Name: loan_status, dtype: int64
```

`According to the confusion matrix in step 3 [Create a LRM w/ Original Imbalanced Data]:`

* Out of the 18,765 loan status's that are healthy (low-risk), the model predicted 18,663 as 
   healthy correctly and 102 as healthy incorrectly. 

* Out of the 619 loan status's that are non-healthy (high-risk), the model 
   predicted 563 as non-healthy correctly and 56 as non-healthy incorrectly.

![Classification Report of Imbalanced DataSet](/Images/confusion_matrix_imbalanced_df.png)

</br>

`To generate a higher accuracy score and have the model catch more mistakes when classifying non-healthy loans, we can oversample the data using the RandomOverSampler module from the imbalanced-learn library, which adds more copies of the minority class (non-healthy loans) to obtain a balanced dataset.`

```
# code
y_oversampled.value_counts()

# output
0    56271
1    56271
Name: loan_status, dtype: int64
```

  * Using the dataset provided by the lending company, I created a Logistic Regression Model fit with the oversampled data that generated an accuracy score of 99%, which turns out to be higher than the model fitted with imbalanced data. The oversampled model performs better due to the dataset being balanced. The models non-healthy loans recall value increased from 0.91 to 0.99 indicating that the model does an exceptional job in catching mistakes such as labeling non-healthy (high-risk) loans as healthy (low-risk).

`According to the confusion matrix in step 3 [Create a LRM w/ Resampled(oversampled) Data]:`

* Out of the 18,765 loan status's that are healthy, the model predicted 18,649 as 
   healthy correctly and 116 as healthy incorrectly. 

* Out of the 619 loan status's that are non-healthy (high-risk), the model 
   predicted 615 as non-healthy correctly and 4 as non-healthy incorrectly.

![Classification Report of Imbalanced DataSet](/Images/confusion_matrix_balanced_df.png)

</br>

## **Results**

</br>

### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Logistic Regression Model fitted with Imbalanced Data: 

</br>

`The Logistic Regression model fitted with the Imbalanced DataSet predicted healthy loans 100% of the time and predicted non-healthy loans 85% of the time.`

</br>

* The model fitted with imbalanced data has a higher possibility of making these mistakes: 

  * a healthy loan (low-risk) is classified as a non-healthy loan (high-risk).
  * a non-healthy loan (high-risk) is classified as a healthy loan (low-risk).

</br>

`According to the models recall scores, the model made 1% of mistakes when predicting healthy loans and made 9% of mistakes when predicted non-healthy loans.`

![Classification Report of Imbalanced DataSet](/Images/classification_report_imbalanced.png)

`The model generated an accuracy score of 95% but could be improved due to the dataset being imbalanced.`

![Accuracy Score of Imbalanced DataSet](/Images/accuracy_score_imbalanced_dataset.png)

</br>

### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Logistic Regression Model fitted with Balanced (oversampled) Data:

</br>

`The Logistic Regression model fitted with the OverSampled DataSet predicted healthy loans 100% of the time and predicted non-healthy loans 84% of the time.`

</br>

* The model fitted with balanced (oversampled) data has a much lower possibility of making these mistakes: 

  * a healthy loan (low-risk) is classified as a non-healthy loan (high-risk).
  * a non-healthy loan (high-risk) is classified as a healthy loan (low-risk).

</br>

`According to the models recall scores, the model made 1% of mistakes when predicting healthy loans and made 1% of mistakes when predicted non-healthy loans.`

![Classification Report of Imbalanced DataSet](/Images/classification_report_balanced.png)

`The model generated an accuracy score of 99% due to the dataset being balanced.`

![Accuracy Score of Imbalanced DataSet](/Images/accuracy_score_balanced_dataset.png)

</br>

## **Summary**

* A lending company might want a model that requires classifying healthy loans and non-healthy loans correctly most of the time: 

    * healthy loans being identified as a non-healthy loan might be more costly for a lending company since it might cause the loss of customers. 
  
    * non-healthy loans being identified as a healthy loan might also be more costly for a lending company due to the loss of funds being provided by the lender.

`The Logistic Regression model fitted with OverSampled data performed much better than the model fitted with Imbalanced data due to the data being balanced and generating a higher accuracy score and a higher recall, indicating that the model will make extremely fewer mistakes when classifying non-healthy loans.`

</br>

`The lending company would most likely want fewer False Positives due to the high possibility of a lender loosing provided funds when classifying non-healthy loans as healthy. The data below is shown in the confusion matrices which indicates how many healthy/non-healthy loans the model predicted correctly/incorrectly.`

* Model fitted with Imbalanced Data: 
  
  * 56 (FALSE POSITIVES) --> The actual value is healthy and the predicted value is non-healthy 


  * 102 (FALSE NEGATIVES) --> The actual value is non-healthy and the predicted value is healthy

</br>

* Model fitted with Balanced Data: 
  
  * 4 (FALSE POSITIVES) --> The actual value is healthy and the predicted value is non-healthy 


  * 116 (FALSE NEGATIVES) --> The actual value is non-healthy and the predicted value is healthy
  
`According to the confusion matrices, the number of False Postives drastically decreases indicating the model will classify healthy & non-healthy loans correctly. Based off of this analysis, I would recommend using Model 2 (Logistic Regression Model fitted with Balanced (oversampled) data.`

---