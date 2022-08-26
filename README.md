Support Vector Regression in Scikit-Learn for Regression

* svr
* support vector regressor 
* python
* feature engine
* scikit optimize
* flask
* nginx
* gunicorn
* docker
* abalone
* auto prices
* computer activity
* heart disease
* white wine quality
* ailerons

This is an implementation of Support Vector Regressor using Scikit-Learn. 

Support Vector Regressor considers the points that are within a decision boundary line and the best fit is the hyperplane that has maximum number of points. 

Points with the least error rate or that are within the Margin of Tolerance define the best fitting model. 

Kernel is the most important feature for an SVR. 

Preprocessing includes missing data imputation, standardization, one-hot encoding etc. For numerical variables, missing values are imputed with the mean and a binary column is added to represent 'missing' flag for missing values. For categorical variable missing values are handled using two ways: when missing values are frequent, impute them with 'missing' label and when missing values are rare, impute them with the most frequent. 

HPT includes choosing the optimal kernel and gamma along with optimal values for degree of the polynomial kernel function (if kernel is 'poly'), total stopping criteria and penalty term C. 

The main programming language is Python. Other tools include Scikit-Learn for main algorithm, feature-engine for preprocessing, Scikit-Optimize for HPT, Flask + Nginx + gunicorn for web service. The web service provides two endpoints- /ping for health check and /infer for predictions in real time.



