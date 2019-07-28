# SparkSentenceClassification

## Intro

Sentence classification using Spark in Java. It uses cosine similarities between sentence word2vec embeddings
(averaged embeddings of words after removing stop words) to calculate similarity with all training 
sentences. Then the class of sentences of with highest average cosine sim is assigned to test sentence.
(Its clear that above is not the best methods both time performance and accuracy wise. See Notes section below.)

## Getting Started

Add the project in JavaIDE as maven project.
Download the csv dataset from 
[Kaggle San Francisco Crime Classification](https://www.kaggle.com/c/sf-crime/data?source=post_page---------------------------).
Extract the csv file in resources folder and set the appropriate path for csv file in Main class.
TRun the Main.

## Notes

The better options with spark are

- Use logistic regression.
- For similarity based method, first use a text search engine library such a lucene to get the best
matching sentence from training set, then use above paradigm on selected sentences. 
 
