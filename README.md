# 🌪️ NLP Disaster Tweets Classification

## 📚 Table of Contents
1. [📝 Introduction](#introduction)
2. [🔍 Problem and Data Description](#problem-and-data-description)
3. [📊 Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [🏗️ Model Architecture](#model-architecture)
5. [📈 Results and Analysis](#results-and-analysis)
6. [🏁 Conclusion](#conclusion)
7. [📚 References](#references)

## 📝 Introduction
This project aims to classify disaster-related tweets using Natural Language Processing techniques. It is a part of the Kaggle competition "Natural Language Processing with Disaster Tweets."

## 🔍 Problem and Data Description
The challenge is to build a machine learning model that predicts whether a given tweet is about a real disaster or not. The dataset consists of 10,000 tweets that have been manually classified. Each sample in the dataset includes the text of the tweet, a keyword from the tweet (if available), and the location the tweet was sent from (if available). [Link to Kaggle Competition](https://kaggle.com/competitions/nlp-getting-started).

## 📊 Exploratory Data Analysis (EDA)

### Loading the Data
Let's start by loading the data and taking a look at the first few rows.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_df.head()
```
### Data Cleaning Procedures
The data cleaning process involves filling in missing values for the 'keyword' and 'location' columns. Missing keywords are filled with 'none', and missing locations are filled with 'unknown'.

```python
# Data cleaning: fill missing values in 'keyword' and 'location' columns
train_df['keyword'].fillna('none', inplace=True)
train_df['location'].fillna('unknown', inplace=True)
```

### Data Visualization
We performed various visualizations to understand the data better:

1. **Distribution of Target Variable:**
   ```python
   # Display the distribution of the target variable
   sns.countplot(x='target', data=train_df)
   plt.title('Distribution of Target Variable')
   plt.show()
   ```
   ![Distribution of Target Variable](./images/distribution_target.png)

   The distribution of the target variable reveals that there are more non-disaster tweets (labeled as 0) compared to disaster-related tweets (labeled as 1). This class imbalance is crucial to consider as it can affect the model's performance. Techniques such as resampling, adjusting class weights, or using specific evaluation metrics will be necessary to handle this imbalance.

2. **Distribution of Tweet Lengths:**
   ```python
   # Display the distribution of tweet lengths
   train_df['text_length'] = train_df['text'].apply(len)
   sns.histplot(train_df['text_length'], bins=50)
   plt.title('Distribution of Tweet Lengths')
   plt.show()
   ```
   ![Distribution of Tweet Lengths](./images/tweet_length.png)

   The distribution of tweet lengths shows that most tweets are around 120 to 140 characters long. This is expected given the character limit on Twitter. Understanding the length of tweets can help in preprocessing, such as padding or truncating tweets to a fixed length when feeding them into the model.

3. **Top 10 Keywords by Frequency:**
   ```python
   # Display the distribution of keyword occurrences
   sns.countplot(y='keyword', data=train_df, order=train_df['keyword'].value_counts().iloc[:10].index)
   plt.title('Top 10 Keywords by Frequency')
   plt.show()
   ```
   ![Top 10 Keywords by Frequency](./images/key_words.png)

   The top 10 keywords by frequency include terms like 'fatalities', 'deluge', 'armageddon', and 'sinking', which are strongly associated with disasters. Keywords can serve as important features for the model, providing context about the content of the tweets.

4. **Top 10 Locations by Frequency:**
   ```python
   # Display the distribution of locations
   sns.countplot(y='location', data=train_df, order=train_df['location'].value_counts().iloc[:10].index)
   plt.title('Top 10 Locations by Frequency')
   plt.show()
   ```
   ![Top 10 Locations by Frequency](path_to_image4.png)

   The distribution of locations shows that most tweets are from the USA, New York, and other prominent locations. Location data can help in understanding regional patterns in disaster reporting and may enhance the model's ability to correctly classify tweets based on where they originate.

### Plan of Analysis
Based on the exploratory data analysis, the plan for analysis includes:

1. **Text Preprocessing:**
   - Tokenization
   - Stopword removal
   - Stemming/Lemmatization
   - Vectorization (e.g., TF-IDF)

2. **Feature Engineering:**
   - Creating features from tweet lengths, keyword, and location data
   - Exploring additional features like sentiment scores

3. **Model Selection and Training:**
   - Experimenting with various models like Logistic Regression, Naive Bayes, and different neural network architectures
   - Hyperparameter tuning and model optimization

4. **Evaluation:**
   - Evaluating models using metrics like F1-score, precision, recall, and accuracy
   - Analyzing results and refining models based on performance

## 🏗️ Model Architecture



## 📈 Results and Analysis


## 🏁 Conclusion


## 📚 References
- Howard, A., devrishi, Culliton, P., & Guo, Y. (2019). Natural Language Processing with Disaster Tweets. Kaggle. Retrieved from [https://kaggle.com/competitions/nlp-getting-started](https://kaggle.com/competitions/nlp-getting-started).