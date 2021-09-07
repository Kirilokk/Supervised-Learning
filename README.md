# Hi, this is my study statistic algorithms i've implemented while DataRoot course(Supervised learning):

+ **Lineral regression with one variable**;
+ Lineral regression with multiple variables;
+ Logistic regression;
+ Polynomial Ridge Regression;


# Lineral regression with one variable 

*Linear regression is used for finding linear relationship between target and one or more predictors.
In this example, it is used to predict kangaroo nasal width *having* its *length*.*


# Project Structure

To keep code organized, it's important to make project structure well.
```

project
    │──── core              - main parts of core. 
    │       ├── optimize.py         - update function.
    │       └── regression.py       - main components of linReg.
    │   
    │   
    │──── data            - data manipulation
    │       ├── dataPreparation.py   - data load and standardization 
    │       └── dataVisualisation.py - test/train visualisation
    │
    │
    └───── interactor             
            └── index.py.            - script to run lineral regression

```


## Data view

Here how our data set looks like:(head(10))

| Index | Height   | Width |
| ------|:--------:| -----:|
| 0     |  609     | 241   |
| 1     |  629     | 222   |
| 2     |  620     | 265   |
| 3     |  564     | 298   |
| 4     |  645     | 256   |
| 5     |  493     | 200   |
| 6     |  606     | 226   |
| 7     |  660     | 240   |
| 8     |  550     | 215   |
| 9     |  480     | 185   |


## Dataset visualisation

![alt text](media/DatasetVisual.png ":)")​

**Important note:**
>  It is important to find **linear relationship(straight line)** between target and one or more predictors


## The results

As a result of Linear regression with one variable algorithm, we got the next relationship, 
where the straight line represents the relation between height and width of kangaroo nasal

![alt text](media/resultPlots.png ":)")​