# Hi, this is my study statistic algorithms i've implemented while DataRoot course(Supervised learning):

+ **Lineral regression with one variable**;
+ Lineral regression with multiple variables;
+ Logistic regression;
+ Polynomial Ridge Regression;


# Lineral regression with one variable 

*Linear regression is used for finding linear relationship between target and one or more predictors.
In this example, it is used to predict kangaroo nasal width *having* its *Length*.*


# Project Structure

To keep code organized, it's important to make project structure well.
```

project
    │──── core                       - main parts of core. 
    │       ├── optimize.py          - update function.
    │       └── regression.py        - main components of linReg.
    │   
    │   
    │──── data                       - data manipulation
    │       ├── dataPreparation.py   - data load and standardization 
    │       └── dataVisualisation.py - test/train visualisation
    │
    │
    └───── interactor             
            └── index.py.            - script to run lineral regression

```


## Data view

Here how our data set looks like:(head(10))

| Index | Length   | Width |
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

```python

# Training set
plt.figure(figsize=(12, 5))
plt.title("Training set")

plt.subplot(1,2,1)
plt.scatter(train_set_x, train_set_y)
x = np.array([min(train_set_x), max(train_set_x)])
theta = d["theta"]
b = d["b"]
y = theta * x + b
plt.plot(x, y)  
plt.axis("tight")
plt.xlabel("Length")
plt.ylabel("Width")
plt.tight_layout()


# Test set

plt.title("Test set")
plt.subplot(1,2,2)
plt.scatter(test_set_x, test_set_y)
x = np.array([min(test_set_x), max(test_set_x)])
theta = d["theta"]
b = d["b"]
y = theta * x + b
plt.plot(x, y)  
plt.axis("tight")
plt.xlabel("Length")
plt.ylabel("Width")
plt.tight_layout()

```  

As a result of Linear regression with one variable algorithm, we got the next relationship, 
where the straight line represents the relation between Length and width of kangaroo nasal

![alt text](media/resultPlots.png ":)")​