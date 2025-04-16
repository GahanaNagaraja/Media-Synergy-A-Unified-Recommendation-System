# MOVIE RECOMMENDATION
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import mean_squared_error


# Read the dataset
df = pd.read_csv('tmdb_5000_movies.csv')

# Preprocess the dataset
df['profitable'] = df.revenue > df.budget
df['profitable'] = df['profitable'].astype(int)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(how="any")

# Log transformation
for covariate in ['budget', 'popularity', 'runtime', 'vote_count', 'revenue']:
    df[covariate] = df[covariate].apply(lambda x: np.log10(1 + x))

# Separate positive revenue movies
positive_revenue_df = df[df["revenue"] > 0]

# Define targets and covariates
regression_target = 'revenue'
classification_target = 'profitable'
all_covariates = ['budget', 'popularity', 'runtime', 'vote_count', 'vote_average']

# User input for genres
genres_input = input("Enter genres (comma-separated): ")
genres = genres_input.split(',')

# Filter movies based on user-input genres
filtered_movies = df[df['genres'].str.contains('|'.join(genres))]

# Extract relevant features
features = filtered_movies[all_covariates].values

# Instantiate models
linear_regression = LinearRegression()
logistic_regression = LogisticRegression()
forest_regression = RandomForestRegressor(max_depth=4, random_state=0)
forest_classifier = RandomForestClassifier(max_depth=4, random_state=0)

# Train models
linear_regression.fit(positive_revenue_df[all_covariates], positive_revenue_df[regression_target])
logistic_regression.fit(positive_revenue_df[all_covariates], positive_revenue_df[classification_target])
forest_regression.fit(positive_revenue_df[all_covariates], positive_revenue_df[regression_target])
forest_classifier.fit(positive_revenue_df[all_covariates], positive_revenue_df[classification_target])

# Predictions
linear_regression_prediction = linear_regression.predict(features)
logistic_regression_prediction = logistic_regression.predict(features)
forest_regression_prediction = forest_regression.predict(features)
forest_classifier_prediction = forest_classifier.predict(features)

# Display results
print('\nMovie Recommended by Highest vote_count on various Algorithm models:')
print("\nRecommendation based on Linear Regression:")
print(filtered_movies[['original_title', 'vote_count']].assign(Linear_Regression_Prediction=linear_regression_prediction).sort_values(by='vote_count', ascending=False).head(1))

print("\nRecommendation based on Logistic Regression:")
print(filtered_movies[['original_title', 'vote_count']].assign(Logistic_Regression_Prediction=logistic_regression_prediction).sort_values(by='vote_count', ascending=False).head(1))

print("\nRecommendation based on Random Forest Regression:")
print(filtered_movies[['original_title', 'vote_count']].assign(Random_Forest_Regression_Prediction=forest_regression_prediction).sort_values(by='vote_count', ascending=False).head(1))

print("\nRecommendation based on Random Forest Classification:")
print(filtered_movies[['original_title', 'vote_count']].assign(Random_Forest_Classification_Prediction=forest_classifier_prediction).sort_values(by='vote_count', ascending=False).head(1))

# Calculate Mean Squared Error and R-squared for Linear Regression
linear_regression_mse = mean_squared_error(positive_revenue_df[regression_target], linear_regression.predict(positive_revenue_df[all_covariates]))
linear_regression_r2 = r2_score(positive_revenue_df[regression_target], linear_regression.predict(positive_revenue_df[all_covariates]))

# Calculate Mean Squared Error and R-squared for Logistic Regression
logistic_regression_mse = mean_squared_error(positive_revenue_df[classification_target], logistic_regression.predict(positive_revenue_df[all_covariates]))
logistic_regression_r2 = r2_score(positive_revenue_df[classification_target], logistic_regression.predict(positive_revenue_df[all_covariates]))

# Calculate Mean Squared Error and R-squared for Random Forest Regression
forest_regression_mse = mean_squared_error(positive_revenue_df[regression_target], forest_regression.predict(positive_revenue_df[all_covariates]))
forest_regression_r2 = r2_score(positive_revenue_df[regression_target], forest_regression.predict(positive_revenue_df[all_covariates]))

# Calculate Mean Squared Error and R-squared for Random Forest Classification
forest_classifier_mse = mean_squared_error(positive_revenue_df[classification_target], forest_classifier.predict(positive_revenue_df[all_covariates]))
forest_classifier_r2 = r2_score(positive_revenue_df[classification_target], forest_classifier.predict(positive_revenue_df[all_covariates]))

# Display results
print("\nPerformance Metrics:")
print(f"Linear Regression MSE: {linear_regression_mse}, R-squared: {linear_regression_r2}")
print(f"Logistic Regression MSE: {logistic_regression_mse}, R-squared: {logistic_regression_r2}")
print(f"Random Forest Regression MSE: {forest_regression_mse}, R-squared: {forest_regression_r2}")
print(f"Random Forest Classification MSE: {forest_classifier_mse}, R-squared: {forest_classifier_r2}")


# MUSIC RECOMMENDATION
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import ast  # Import the ast module for literal_eval

# Load the dataset
data = pd.read_csv('data.csv')

# Convert the 'artists' column from string representation of a list to an actual list
data['artists'] = data['artists'].apply(ast.literal_eval)

# Explode the lists in the 'artists' column to separate rows for each artist
data_exploded = data.explode('artists')

# Group by song and calculate the mean popularity for each song
song_popularity = data_exploded.groupby('name')['popularity'].mean().reset_index()

# Define the features (X) and the target variable (y)
feature_names = ['valence', 'year', 'acousticness', 'danceability',
                 'duration_ms', 'energy', 'explicit', 'instrumentalness', 'key', 'liveness',
                 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']
X = data_exploded.groupby('name')[feature_names].mean()  # Take the mean of features for each song
y = data_exploded.groupby('name')['popularity'].mean()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Now, let's recommend one song based on the linear regression model
# Sort songs by predicted popularity in descending order
song_predictions = pd.DataFrame({'Song': X.index, 'Predicted Popularity': model.predict(X)})
sorted_songs = song_predictions.sort_values(by='Predicted Popularity', ascending=False)

# Select the top row as the recommended song
recommended_song = sorted_songs.head(1)

# Merge with the original dataset to get song names
recommended_song = pd.merge(recommended_song, song_popularity, left_on='Song', right_on='name', how='left')

# Display the recommended song along with its name and predicted popularity
print('\nMusic Recommended by Highest Predicted Popularity:')
print(recommended_song[['Song', 'name', 'Predicted Popularity', 'popularity']])


# BOOK RECOMMENDATION
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
books_data = pd.read_csv('books.csv')

# Explore and preprocess the data if needed
# For simplicity, assuming the 'average_rating' column is already available and cleaned


# Define features (X) and target variable (y)
feature_names = ['books_count', 'original_publication_year', 'ratings_count', 'work_ratings_count', 'work_text_reviews_count']
X = books_data[feature_names]
y = books_data['average_rating']

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Now, let's recommend books based on the linear regression model
# Sort books by predicted average rating in descending order
book_predictions = pd.DataFrame({'Book Title': books_data['title'], 'Predicted Average Rating': model.predict(X)})
sorted_books = book_predictions.sort_values(by='Predicted Average Rating', ascending=False)

# Display the recommended books along with their titles and predicted average ratings
# print('\nBooks Recommended by Highest Predicted Average Rating:')
# print(sorted_books[['Book Title', 'Predicted Average Rating']])

# ... (previous code)

# Now, let's recommend one book based on the linear regression model
# Sort books by predicted average rating in descending order
book_predictions = pd.DataFrame({'Book Title': books_data['title'], 'Predicted Average Rating': model.predict(X)})
sorted_books = book_predictions.sort_values(by='Predicted Average Rating', ascending=False)

# Select the top row as the recommended book
recommended_book = sorted_books.head(1)

# Display the recommended book along with its title and predicted average rating
print('\nBook Recommended by Highest Predicted Average Rating:')
print(recommended_book[['Book Title', 'Predicted Average Rating']])


