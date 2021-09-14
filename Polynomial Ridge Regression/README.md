# Hi, this is my study statistic algorithms i've implemented while DataRoot course(Supervised learning):

+ Lineral regression with one variable;
+ Lineral regression with multiple variables;
+ Logistic regression;
+ ### **Polynomial Ridge Regression;**


# Polynomial Ridge Regression  

Not everything we can visualise with a straight line. Sometimes the prediction may be unpredictable. In case of this, our solution is **Polynomial Ridge Regression**.<br/>
In this type of regression the input parameters are used to create higher nth degree polynomials on which a model is trained for prediction. As a result, we will have a curve representing our dataset.


As for *Ridge regression*, it's quite useful regularization technique used to adress over-fitting(when a statistical model fits exactly against its training data and againts test examples fails).<br/>
It is very similar to Linear Regression only that it differs in cost function. Here we have some **penalty term(lambda)** <br/>The lambda parameter controls the shrinkage of the term. If it’s set to 0 then the entire equation becomes like normal Linear Regression curve and high values of lambda ensure the ridge regression to overfit the data.


# Ridge regression formula used in algorithm:

![alt text](media/Ridge_regression_formula.gif "^_^")​


## Data view

Here how our data set looks like:(head())


## Data view

Here how our data set looks like:(head())

 |Days(0 to 1) | Temperature|
 |-------------|-----------:|
 |	0.00273224|	0.1  |  
 |	0.00546448|	-4.5 |
 |	0.00819672|	-6.3 |
 |	0.01092896|	-9.6 |
 |	0.01366120|	-9.9|	

## Brief info

**Our Day feature is already normalized to 0-1 range, so next, we will multiply it by 366 to restore the correct day of the year.**

> In this example, out goal is to predict annual temperature, visualise polynomial dependence our data.

## Dataset visualisation

Here is shown temperature dependency from days in year:

![alt text](media/data_plot.png ":)")​


## The results

As we can see, our model fits well the hypothesis function to the data. Despite having high-degree polynomials, we prevented overfitting by using the **L2 Regularization(Ridge)** - method for penalizing high magnitudes of parameters estimates. Also we implemented *Polynomial Ridge Regression* model with OOP in mind.

![alt text](media/result.png ":)")​