import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
data = pd.read_csv('lung.csv')

# Preprocess the data
X = data.drop('survival_time', axis=1)
y = data['survival_time']

# Perform one-hot encoding on the categorical variables
categorical_cols = ['gender', 'smoking_history', 'treatment_type']
encoder = OneHotEncoder(sparse=False)
X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]))
X_encoded.columns = encoder.get_feature_names(categorical_cols)

# Drop the original categorical columns from X and concatenate the encoded columns
X = pd.concat([X.drop(categorical_cols, axis=1), X_encoded], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, predictions)
print('Mean Squared Error:', mse)

# Provide a sample output
sample_patient = X.sample(1)  # Choose a random row from the dataset
survival_time = model.predict(sample_patient)
print('Predicted Survival Time for the Sample Patient:', survival_time)
