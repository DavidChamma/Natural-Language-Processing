# Incremental Skip-gram (Word2Vec)

The goal of this project is to implement a skip-gram model with negative-sampling from scratch. We implemented the incremental method to gain in efficiency. We used a log-likelihood function based on the conditional probability of the context knowing the word, parametrized by the embedding of the words in our vocabulary.
The parameters are updated by minimizing minus the log-likelihood using Gradient Ascent.

Input : corpus of words (1-billion-word-language-modeling-benchmark-r13output.tar.gz)
Output: embedding of the words in the corpus

## Prerequisites

Libraries used:
argparse
pandas as pd
numpy as np
nltk # Used to remove stopwords from the original corpus and to tokenize the corpus
math # Used to define the sigmoid
string  # Used to get all punctuation
pickle  # Used to save and load embeddings


## Preprocessing data

1. Text to sentences
Uploading data which can be in different types (tar.gz in our case). Then splitting sentences '\n'. And finally splitting words using whitespaces after removing punctuation, non-alpha and stopwords.

2. Rare words pruning	
Removing  words that appears less than minCount in the corpus. 

3. High frequency words removing
Remove words that occurs more than a fixed threshold in the corpus.
We haven't been using this method in the final model due to worse results than by using stopwords.



## Incremental Skip Gram model

For each iteration, create 1 positive example and k = negativeRate negative examples
1. Positive pair 
Fixing a window size winSize. Generating positive pairs (target word, context word). 

2. Negative pairs
For each positive pair, choose k = negativeRate of random words to take from the corpus to create (target word, random word).


## Train the model

### Method : Incremental method in order to actualise the embedding of the words each time we create 1 positive examples and the related negative examples
For a corpus containing V unique words, we will compute a matrix of V rows and nEmbed columns stored in a dictionnary to have the word (or string) as key.

We implemented the algorithm as follow:
1. Initializing coeff (dictionnary of embedding to update) with a Uniform distribution (-0.005, 0.005). Avoiding vector of zeros or the gradient will remain null. 
2. Choosing number of epochs (epochs = 5 in our case).
3. Compute the gradient of the loss derived from the log-likelihood objective function (formula given by 'Yoav Goldberg' and 'Omer Levy'[1])
4. Update directly embeddings after the generation of each positive and its related k = negativeRate negative examples.


## Running the model

type of Command line to execute : 
```
python skipGram.py --news/news.en-00001-of-00100.txt --model news
```
The command uses the news.en-00001-of-00100.txt as training set and saves the word embeddings in news file.

Example :
news.en-00001-of-00100.txt  file containing :
- 50 000 sentences
- 1 094 214 words
- 5 576 574 positive pairs
- 27 882 870 negative pairs
- 14 033 unique words 

## Testing the model

We used the cosine distance for the similarity. The command to type is:

```
python skipGram.py --text data/news.csv --model news --test
```
Example of outputs:
We computed the similarity between the word "president" and all the words of the corpus, and printed its most similar words.
For 10000 sentences and 5 epochs we get:
```
Similar words for president :
- vice : 0.41
- criticised : 0.40
- barack : 0.35
- rosie : 0.34
- chechnya : 0.32
- socialist : 0.32
- egyptian : 0.32
- bush : 0.32
- mahmoud : 0.31
- authentic : 0.30
```

10 most similar words to "city":
```
Similar words for city :
- nearby : 0.41
- move : 0.38
- representatives : 0.37
- homes : 0.36
- owning : 0.36
- ny : 0.35
- sands : 0.34
- example : 0.33
- bronx : 0.33
- button : 0.33
```

## References
[1] Yoav Goldberg and Omer Levy _word2vec Explained: Deriving Mikolov et al.’s Negative-Sampling Word-Embedding Method_. 2014
[2] NobuhiroKaji and HayatoKobayash, IncrementalSkip-gramModelwithNegativeSampling. 2015