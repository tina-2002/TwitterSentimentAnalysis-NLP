# Hate Speech Detection in Tweets

## Overview

This project aims to classify tweets as racist/sexist (hate speech) or non-racist/non-sexist. Using Natural Language Processing (NLP) techniques, the objective is to detect and classify harmful sentiments such as racism or sexism in tweets.

The dataset consists of 31,962 labeled tweets, where label `1` denotes a tweet is racist/sexist, and label `0` denotes it is not. The task is to train a machine learning model to predict whether a tweet contains hate speech or not.

## Dataset

The dataset used for this project is `Twitter Sentiments.csv`, which contains the following columns:

- `id`: A unique identifier for each tweet.
- `label`: The label indicating whether the tweet is racist/sexist (`1`) or not (`0`).
- `tweet`: The text of the tweet.

## Objective

- **Goal**: Detect and classify tweets with racist/sexist content.
- **Input**: Raw tweets.
- **Output**: Binary classification (1 for racist/sexist, 0 for non-racist/non-sexist).

## Features

1. **Data Preprocessing**: Clean the tweet data by removing unwanted patterns, special characters, short words, and Twitter handles. The text is tokenized, and stemming is applied.
2. **Exploratory Data Analysis (EDA)**: 
   - Visualize the most frequent words in both positive (non-hate) and negative (hate) tweets.
   - Extract and visualize common hashtags used in both classes.
3. **Feature Extraction**: Use the Bag-of-Words (BoW) model to convert text data into numerical vectors.
4. **Model Training**: Train a Logistic Regression model to classify tweets into hate speech or not.
5. **Model Evaluation**: Evaluate the model using metrics like F1-score and accuracy.

## Libraries Used

- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical computing.
- **matplotlib**: For visualizations.
- **seaborn**: For statistical data visualization.
- **nltk**: For natural language processing (tokenization, stemming).
- **sklearn**: For machine learning (feature extraction, model training, and evaluation).
- **wordcloud**: For generating word clouds to visualize frequent words.

## Steps

### 1. Preprocessing

The following steps are applied to clean the tweet text:
- Remove Twitter handles (e.g., `@user`).
- Remove special characters, numbers, and punctuation.
- Remove short words (less than 4 characters).
- Tokenize the text into individual words.
- Apply stemming to reduce words to their base form.

### 2. Exploratory Data Analysis (EDA)

- Visualize frequent words using word clouds.
- Extract and analyze hashtags used in both racist/sexist and non-racist/sexist tweets.

### 3. Feature Extraction

- Use **CountVectorizer** to convert text data into a Bag-of-Words (BoW) model with a set of 1000 most frequent words.
- Split the data into training and testing sets.

### 4. Model Training

- A **Logistic Regression** model is trained on the training dataset to classify tweets.
- The model's performance is evaluated using **accuracy** and **F1-score**.

### 5. Evaluation

- The F1-score and accuracy metrics are computed for both the initial model and an improved model using probability thresholds.


## Results

- **F1-score**: 0.554
- **Accuracy**: 94.3%

## Visualizations

- Word cloud visualizations for frequent words in both non-hate and hate tweets.
- Bar charts for the most frequent hashtags in both classes of tweets.

## Conclusion

This project demonstrates the power of machine learning and NLP techniques to detect hate speech in text. The Logistic Regression model provides a good baseline, achieving high accuracy in classifying tweets.

