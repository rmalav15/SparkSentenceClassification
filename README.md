# SparkSentenceClassification

## Intro

Sentence classification using Spark in Java. 
For each test sentence, the cosine similarity is calculated with all training sentences.
Averaged word2vec embeddings of words (after removing stop words) is used as sentence embedding. 
The class of sentences with highest average cosine similarity is assigned to test sentence.
(Its clear that above is not the best methods both time performance and accuracy wise. See Notes section below.)

## Getting Started

Add the project in JavaIDE as maven project. I have used 
[Kaggle San Francisco Crime Classification](https://www.kaggle.com/c/sf-crime/data?source=post_page---------------------------)
dataset.
Extract the csv file in resources folder and set the appropriate path in Main class.
Run the main method.

## Notes

The better options with spark are

- Use logistic regression.
- For similarity based method, first use a text search engine library such a lucene to get the best
matching sentence from training set, then use above paradigm on selected sentences. 
 
