#!/usr/bin/env python
# coding: utf-8

# In[265]:


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# In[266]:


housing=pd.read_csv("housing.csv")


# In[267]:


housing.head()


# In[268]:


housing.info()


# In[269]:


#There are 20,640 instances in the dataset,but in total_bed
#rooms attribute has only 20,433 that means other 207 value are not null


# In[270]:


housing.isnull().sum()


# In[271]:


#All attributes are numerical, except the ocean_proximity field.
housing["ocean_proximity"].value_counts()


# In[272]:


housing.describe()#that the null values are ignored


# In[273]:


# plot a histogram for each numerical attribute
housing.hist(bins=50, figsize=(20,20))


# In[274]:


housing.hist(bins=50, figsize=(20,20))
plt.show()


# In[275]:


# for test and trian split we used sklearn.model
from sklearn.model_selection import train_test_split


# In[276]:


#Create a Test Set 20% and train 80%

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

print(len(train_set))
print(len(test_set))


# In[277]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# In[278]:


train_set, test_set


# In[279]:


#Histogram of income categories
housing["income_cat"]=pd.cut(housing["median_income"],# median_income is col in dataset
                             bins=[0.,1.5,3.0,4.5,6.,np.inf],
                             labels=[1,2,3,4,5])


# In[280]:


housing["income_cat"].hist()


# In[281]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
 strat_train_set = housing.loc[train_index]
 strat_test_set = housing.loc[test_index]

strat_test_set["income_cat"].value_counts() / len(strat_test_set)


# In[282]:


# income_cat attribute so the data is back to its original state:
for set_ in (strat_train_set, strat_test_set):
 set_.drop("income_cat", axis=1, inplace=True)


# In[283]:


#Discover and Visualize the Data to Gain Insights
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(train_set.describe())  # Summary statistics for numerical columns


# In[284]:


train_set.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, c='blue', label='Training Set') 
test_set.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, c='red', label='Test Set')
plt.title('Longitude vs Latitude')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.show() 


# In[285]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
 s=housing["population"]/100, label="population", figsize=(10,7),
 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()


# In[286]:


from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",
 "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))


# In[287]:


housing.plot(kind="scatter", x="median_income", y="median_house_value",
 alpha=0.1)


# In[288]:


#Prepare the Data for Machine Learning Algorithms
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


# In[289]:


#Data Cleaning drop the null value and fill with median
housing.dropna(subset=["total_bedrooms"])
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median, inplace=True)


# In[290]:


# data using sklearn method
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)


# In[291]:


imputer.statistics_


# In[292]:


housing_num.median().values


# In[293]:


#imputer to transform the training set by replacing missing values by the learned medians:
X = imputer.transform(housing_num)


# In[294]:


#The result is a plain NumPy array containing the transformed features. If you want to
#put it back into a Pandas DataFrame, it’s simple:
    
housing_tr = pd.DataFrame(X, columns=housing_num.columns)


# In[295]:


#Handling Text and Categorical Attributes
housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)


# In[296]:


#text to numbers we used Scikit-Learn’s Ordina lEncoder class
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]



# In[297]:


ordinal_encoder.categories_


# In[298]:


#categories 0 and 4 are clearly more similar than categories 0 and 1
#To fix this issue, a common solution is to create one binary attribute per category:
#one attribute equal to 1 when the category
#is “<1H OCEAN” (and 0 otherwise), another attribute equal to 1 when the category is
#“INLAND” (and 0 otherwise), and so on. This is called one-hot encoding,


# In[299]:


from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot


# In[300]:


housing_cat_1hot.toarray()


# In[301]:


cat_encoder.categories_


# In[302]:


#Custom Transformers: combining specific attributes.
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
        
    def fit(self, X, y=None):
        return self  # nothing else to do
    
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

# Define numerical columns
housing_num = housing.drop("ocean_proximity", axis=1)


# In[303]:


# Define numerical pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])


# In[304]:


#Transformation Pipelines
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
 ('imputer', SimpleImputer(strategy="median")),
 ('attribs_adder', CombinedAttributesAdder()),
 ('std_scaler', StandardScaler()),
 ])
housing_num_tr = num_pipeline.fit_transform(housing_num)


# In[305]:


#Select and Train a Model
#Training and Evaluating on the Training Set

# Define column names
num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']

# Create the full pipeline
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', OneHotEncoder(), cat_attribs),
])

# Transform the data
housing_prepared = full_pipeline.fit_transform(housing)

print(housing_prepared)


# In[306]:


# Select and Train a Model

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# In[308]:


some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))


# In[309]:


print("Labels:", list(some_labels))


# In[310]:


#measure this regression model’s RMSE on the whole train‐ing set using Scikit-Learn’s mean_squared_error function:
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[311]:


# DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)


# In[312]:


housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# In[313]:


#Better Evaluation Using Cross-Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
 scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


# In[314]:


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)


# In[315]:


#Let’s compute the same scores for the Linear Regression model just to be sure:

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


# Decision Tree model is overfitting so badly that it performs worse
# than the Linear Regression model.

# RandomForestRegressor: Building a model on top of many other models is called Ensemble Learning

# In[316]:


# Train the RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()

forest_reg.fit(housing_prepared, housing_labels)


# In[317]:


# Make predictions
housing_predictions = forest_reg.predict(housing_prepared)

# Calculate RMSE
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)

print(forest_rmse)


# In[318]:


display_scores(forest_rmse)


# In[319]:


#Grid Search
from sklearn.model_selection import GridSearchCV
param_grid = [
 {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
 {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
 ]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
 scoring='neg_mean_squared_error',
return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)


# In[320]:


grid_search.best_params_


# In[ ]:


grid_search.best_estimator


# In[322]:


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# In[323]:


#Analyze the Best Models and Their Errors
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances


# In[324]:


#display these importance scores next to their corresponding attribute names:
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)


# In[326]:


#Evaluate Your System on the Test Set
# Retrieve the best estimator
final_model = grid_search.best_estimator_

# Prepare the test data
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)

# Make predictions on the test data
final_predictions = final_model.predict(X_test_prepared)

# Calculate the final RMSE
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print("Final RMSE:", final_rmse)


# In[328]:


from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
loc=squared_errors.mean(),
scale=stats.sem(squared_errors)))


# In[329]:


#Launch, Monitor, and Maintain Your System


# In[ ]:




