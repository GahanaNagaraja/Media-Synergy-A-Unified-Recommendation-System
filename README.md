# Media-Synergy-A-Unified-Recommendation-System
A unified recommendation system delivering personalized suggestions across music, movies, and books using K-means clustering, collaborative filtering, and regression models to tackle the long-tail and serendipity challenges.

## Project Overview

In the digital age, users are overwhelmed with an abundance of media content across various platforms—music, movies, and books. Traditional recommendation systems often operate in silos, limiting the user’s experience to a single content domain. This project introduces a Unified Recommendation System, aiming to seamlessly integrate multiple media types into a single, intelligent platform. By leveraging machine learning algorithms, the system personalizes suggestions, enhances content discoverability, and solves common issues like the long-tail and serendipity problems.

## Problem Statement

Most existing recommendation systems are domain-specific, focusing on either music, movies, or books individually. This fragmentation leads to:

- A disjointed user experience across platforms

- Difficulty in discovering niche or less popular content (long-tail problem)

- Lack of novelty in recommendations (serendipity problem)

## Objective

- To build a unified recommendation engine that:

- Recommends diverse content types from a single interface

- Uses clustering, regression, and collaborative filtering for personalized suggestions

- Enhances user satisfaction by balancing relevance and novelty

## Features

- Music recommendation using K-Means Clustering

- Movie recommendation using Linear Regression, Logistic Regression, and Random Forest

- Book recommendation using Collaborative Filtering

- Preprocessing, feature selection, and dimensionality reduction

- Popularity and similarity-based ranking

## Datasets

Sourced from Kaggle:

- Spotify Music Dataset (audio features, genres, artists)

- TMDb Movie Dataset (budget, ratings, genre, revenue)

- Books Dataset (ratings, title, author, genres)

## Technologies Used

- Python

- Pandas, NumPy

- Scikit-learn

- Seaborn, Matplotlib, Plotly

- Yellowbrick

- SMOTE (for class imbalance handling)

## Implementation Overview

### Music Recommendation

- Clustering genres and songs based on audio features

- t-SNE & PCA used for visualization

- Personalized genre/song suggestions using cluster-based filtering

### Movie Recommendation

- Regression models trained to predict vote counts and revenue

- User selects a genre to get top-rated movie suggestions

### Book Recommendation

- Hybrid of popularity-based and collaborative filtering

- Recommends based on user similarity and book ratings

## Evaluation Metrics

- R² Score and Mean Squared Error (for regression models)

- Visual cluster evaluation (K-Means)

- Top-N recommendation validation (collaborative filtering)

## Conclusion

The Unified Recommendation System successfully demonstrates the feasibility of integrating different media types into a single, cohesive recommendation platform. By combining clustering techniques for music, regression-based models for movies, and collaborative filtering for books, the system addresses core challenges in content recommendation—relevance, diversity, and novelty.

The modular design allows for scalability and can be extended to include additional content types such as podcasts or TV shows. With further development, including UI integration and real-time user feedback, this system can evolve into a full-scale personalized content discovery tool. The project not only advances technical competency in machine learning and data processing but also showcases the potential for creating cross-domain intelligent systems that enrich user engagement.

## Future Scope

- Improve cold-start handling using hybrid models

- Integrate with user authentication systems

- Deploy as a web-based app with interactive UI

- Extend to include more media domains (podcasts, news)

## Authors

1. Gahana Nagaraja

2. Ashwini Ramesh Benni

3. Vaishnavi Rajendra Dhotargavi
