# Movie-success-rate-and-budget-prediction-using-Machine-Learning
## Project outline:

This Project basically concentrates on predicting the upcoming movies budget and their success using Machine Learning Algorithms. For producers, studios, and directors, projecting the budget and box office performance of prospective films is a vital undertaking in the fast-paced and fiercely competitive film industry. This study develops a useful model that can reduce the likelihood of failure, give stakeholders confidence, and provide a clear forecast of success. The projection takes into account several factors such as the budget, cast, director, producer, gross, genres, language, ratings, duration, movie title and release year etc... By using many attributes from the dataset, the revenue and budget will be predicted. Although it yields suitable results, it can yet be improved in terms of accuracy and efficiency.

## Methodology:

Finding a sample and analytically-worthy movie data dataset is
the first stage.
Genre, language, and details about the actors and directors
engaged in the film productions are examples of general pre-
production information that must be included in this type of
material.
This process involves training the model by using various
Machine learning techniques and algorithms.
Thus predicts the gross, revenue and success of the upcoming
movies.


## Sample Dataset:
<img width="848" alt="image" src="https://github.com/Lakshmipriya-P-AI/Movie-success-rate-and-budget-prediction-using-Machine-Learning-/assets/93427923/5fd60594-20d9-44ab-b765-18039a900ca5">

# Program
## Predicting gross revenue and budget
### Predicting gross revenue
```
X_gross = df.drop(['gross', 'budget'], axis=1)
y_gross = df['gross']
```
### Predicting budget
```
X_budget = df.drop(['gross', 'budget'], axis=1)
y_budget = df['budget']
```
### Split the data for gross revenue prediction
```
X_train_gross, X_test_gross, y_train_gross, y_test_gross = train_test_split(X_gross, y_gross, test_size=0.2, random_state=42)
```
### Split the data for budget prediction
```
X_train_budget, X_test_budget, y_train_budget, y_test_budget = train_test_split(X_budget, y_budget, test_size=0.2, random_state=42)
```
### Linear regression for gross revenue
```
linear_model_gross = LinearRegression()
linear_model_gross.fit(X_train_gross, y_train_gross)
```

### Linear regression for budget
```
linear_model_budget = LinearRegression()
linear_model_budget.fit(X_train_budget, y_train_budget)
```
### Evaluate the linear regression models
```
y_pred_gross = linear_model_gross.predict(X_test_gross)
mse_gross = mean_squared_error(y_test_gross, y_pred_gross)

y_pred_budget = linear_model_budget.predict(X_test_budget)
mse_budget = mean_squared_error(y_test_budget, y_pred_budget)

print(f"Linear Regression - Gross Revenue Mean Squared Error: {mse_gross}")
print(f"Linear Regression - Budget Mean Squared Error: {mse_budget}")
```
### Predict gross revenue using the trained linear regression model
```
sample_input_gross = X_test_gross.head(1)  # Assuming you want to predict the gross for the first test sample

predicted_gross = linear_model_gross.predict(sample_input_gross)

print(f"Predicted Gross Revenue: {predicted_gross[0]}")
```
## Success Prediction

### Define the criterion for success (e.g., movies with gross revenue above a certain threshold)
```
success_threshold = 100000000  # Adjust this threshold based on your definition of success
```
### Create a binary target variable indicating success (1) or not (0)
```
data['success'] = (data['gross'] > success_threshold).astype(int)
```
### Select relevant columns for predicting success
```
selected_columns = ['num_critic_for_reviews', 'duration', 'director_facebook_likes',
                     'actor_3_facebook_likes', 'actor_1_facebook_likes', 'num_voted_users',
                     'cast_total_facebook_likes', 'facenumber_in_poster', 'num_user_for_reviews',
                     'imdb_score', 'budget', 'title_year', 'actor_2_facebook_likes', 'movie_facebook_likes']
```
### Drop rows with missing values in the selected columns
```
df = data[selected_columns + ['success']].dropna()
```
### Define features and target variable
```
X = df.drop('success', axis=1)
y = df['success']
```
### Split the data into training and testing sets
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
### Create and train a logistic regression model
```
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)
```
### Make predictions on the test set
```
y_pred = logreg_model.predict(X_test)
```
### Evaluate the model
```
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
```
# Output
![de1b41a4-08a1-4a07-a104-1ea09dfd37e6](https://github.com/Lakshmipriya-P-AI/Movie-success-rate-and-budget-prediction-using-Machine-Learning-/assets/93427923/b5ad2cc0-e473-4dc2-8345-2b14a437c85e)
![139b9a82-045d-423e-983f-eb670f903b77](https://github.com/Lakshmipriya-P-AI/Movie-success-rate-and-budget-prediction-using-Machine-Learning-/assets/93427923/d6e7b49d-d717-4809-8b3a-7952fc218ad6)
![563149f6-cfa2-44a7-a1f8-71c8e9ded9c9](https://github.com/Lakshmipriya-P-AI/Movie-success-rate-and-budget-prediction-using-Machine-Learning-/assets/93427923/503992d7-7813-43b0-9e64-446e40845474)
![image](https://github.com/Lakshmipriya-P-AI/Movie-success-rate-and-budget-prediction-using-Machine-Learning-/assets/93427923/de06d9c1-6671-493b-abc8-bdb56a25cd77)

![image](https://github.com/Lakshmipriya-P-AI/Movie-success-rate-and-budget-prediction-using-Machine-Learning-/assets/93427923/11ab2060-8bcf-439d-ae2e-9d39f6ceebe9)

# flow of the project
![image](https://github.com/Lakshmipriya-P-AI/Movie-success-rate-and-budget-prediction-using-Machine-Learning-/assets/93427923/dde602fc-f537-4990-ba55-0562e2e51d0a)

Firstly, The data set is collected and preprocessed. Using decision tree, The most important five factors are considered to predict the outcome.
The gross and budget value are predicted using linear regression.
The success rate is determined using the logistic regression.

# Result
Finally, gross and budget predictions are made using linear regression. 
The accuracy of the success rate is determined using logistic regression.


   
