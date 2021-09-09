# Hi, this is my study statistic algorithms i've implemented while DataRoot course(Supervised learning):

+ Lineral regression with one variable;
+ **Lineral regression with multiple variables**;
+ Logistic regression;
+ Polynomial Ridge Regression;


# Lineral regression with multiple variables

*Linear regression is used for finding linear relationship between target and one or more predictors.*<br/>
LR with with multiple variables is more vectorized that with one variable where we used scalar values mainly.<br/>
*In this example, it is used to predict *Boston House Prices* having its additional information(ll be shown next)*

# Project Structure

To keep code organized, it's important to make project structure well.
>Lineral regression with multiple variables *project structure* is the same as with one variable

```

project
    │──── core              - main parts of core. 
    │       ├── optimize.py         - update function.
    │       └── regression.py       - main components of linReg.
    │   
    │   
    │──── data            - data manipulation
    │       ├── dataPreparation.py   - data load and standardization.
    │       └── dataVisualisation.py - test/train visualisation.
    │
    │
    └───── interactor             
            └── index.py.            - script to run lineral regression.

```


## Data view

Here how our data set looks like:(head())

 |	CRIM   | ZN  |INDUS  |CHAS |NOX |RM |AGE  |DIS  |RAD  |TAX  |PTRATIO |B     |LSTAT |
 |---------|:---:|:-----:|:---:|:--:|:-:|:---:|:---:|:---:|:---:|:------:|:----:|-----:|
0|	0.00632|	18.0|	2.31|	0.0|	0.538|	6.575|	65.2|	4.0900|	1.0|  296.0 |15.3   |396.90|	4.98|
1|	0.02731|	0.0 |	7.07|	0.0|	0.469|	6.421|	78.9|	4.9671|	2.0|  242.0	|17.8	|396.90|	9.14|
2|	0.02729|	0.0 |	7.07|	0.0|	0.469|	7.185|	61.1|	4.9671|	2.0|  242.0	|17.8	|392.83|	4.03|
3|	0.03237|	0.0 |	2.18|	0.0|	0.458|	6.998|	45.8|	6.0622|	3.0|  222.0	|18.7	|394.63|	2.94|
4|	0.06905|	0.0|	2.18|	0.0|	0.458|	7.147|	54.2|	6.0622|	3.0|  222.0	|18.7	|396.90|	5.33|


## House features

```
Attribute Information (in order):
    - CRIM     per capita crime rate by town
    - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
    - INDUS    proportion of non-retail business acres per town
    - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    - NOX      nitric oxides concentration (parts per 10 million)
    - RM       average number of rooms per dwelling
    - AGE      proportion of owner-occupied units built prior to 1940
    - DIS      weighted distances to five Boston employment centres
    - RAD      index of accessibility to radial highways
    - TAX      full-value property-tax rate per $10,000
    - PTRATIO  pupil-teacher ratio by town
    - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    - LSTAT    % lower status of the population
    - MEDV     Median value of owner-occupied homes in $1000's

```
## Dataset visualisation

Here is shown house price dependency from house feature shown above ^

![alt text](media/DataVisualisation.png ":)")​

**Important note:**
>  It is important to find **linear relationship(straight line)** between target and one or more predictors


## The results

As a result of Linear regression with multiple variables algorithm, we got quite nice linear dependecy between predicted and true values
where the straight line represents the relation between price and number of features.

![alt text](media/Result.png ":)")​