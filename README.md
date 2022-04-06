## Language Detection

The project is to find languages in which the text was written.
The language list was shorten to just Russian, Ukrainian, Belarusian, Kazakh, Azerbaijani, Armenian, Georgian, Hebrew, 
English and German. \
One of the most common and simple approaches is TFIDF encodings with (1,5) N-Grams and Logistic Regression, which was implemented here 
as a baseline. \
Nevertheless, transformers are known to be very good at language understanding, so as a second approach 
"bert-base-multilingual-cased" model was trained and compared to logreg one. \
Repository contents: 
##### _root_
 - load_checkpoints.sh - bash file to download model weights from google drive (_gdown_ package should be installed)
##### _scripts folder_
- initializer.py - some preparations for future work like dataset downloading, some parameter initializations.
- metrics.py - class for metric computations (accuracy, f1).
- ngram_model.py - class with logreg model (with which you can train, test your model and make sentence predictions).
- dataset.py - class for dataset preparation for Bert model.
- bert_model.py - class with bert model (with which you can train, test your model and make sentence predictions).
- train.py - script to train your model (logreg or Bert), the training configurations and model type are set in config.ini file.
- predict.py - with it you can get languages predictions (one or many) for an input sentence (from terminal).
- Experiments.ipynb - jupyter notebook with experiments were you can train/test models (alternative to train.py/predict.py) 
and see some visualization.
##### _data folder_
- essential dataset parts are held in data folder.
##### _results folder_
- resulting checkpoints are saved to results folder.

### Dataset description

Some description you can see at dataset repo (https://github.com/huggingface/datasets/tree/master/datasets/wili_2018). \
Dataset contains sentence examples for 235 languages (500 samples for each language). It was formed from Wikipedia.

### Results

As was mentioned above, I used 2 methods for detecting languages in the text. Both methods can predict one or many languages, 
depending on threshold. 

1) Char-level ngram logistic regression model with tfidf encodings. Showed high accuracy level (0.9876 on test dataset). 
It learns char ngrams (from one grams to five grams) from scratch, so it could be easily broadcast to other languages. 
Moreover, this approach showed good generalisation ability (to predict language in a data sample with only one language) even on not so big datasets.

2) Bert with its own tokenizer, which splits words on meaningful (bpe) subparts. These model was a bit worse, 
but it could be easily explained, Bert is multilingual, so it knows much extra infomation. 
Moreover, we retrained only classification layer, encoding layers were freezed as our dataset and computer abilities 
are not enough for appropriate retraining of the whole Bert.
Nevertheless, accuracy is still high (0.98 on test set).

|    Best F1-scores:   |N-Gram| mBERT| 
| ---------------|:-----------:| :-----------:|
| **WILI-2018 test**|    __98.76%__   |  98.16%|

In this table accuracy is high, but in test dataset were presented only sentences written in only one language. 
Experiments with sentences, containing 2 languages showed very bad ability of such models to predict both labels, 
but it was expected behavior. Training dataset contains only single-language samples and models were trained to predict only one label well.
So, models are quite confused when they get something different. \
We can overcome these issues by changing model architectures and logic, changing dataset to have multilanguage samples.\
We can use perplexity. Our models are good at predicting one language, so we can choose the most probable language and measure perplexity of the sentence. 
If perplexity is high then it is possible that sentence contains several languages.
Here we can user rule-based approaches. For example, we can see the characters from which the word (sentence) is constructed, 
then compare these list with alphabets of all languages. If we found only one appropriate alphabet, then we do not need anything else, we reached the goal.
If several alphabets are appropriate for the sentence then we can use some classifier to predict probabilities of languages from defined list.
The list of languages to predict is restricted, so may be classification will be better. (and output several most probable languages, using some threshold)
To find the place, where the language is changed we can use bigrams and unigrams of words in a sentences and see the perplexity for predicted languages (if the are several for the sentence), 
in the cases were for 1st word we have low perplexity for language 1, for 2nd word we have low perplexity for language 2 and possibly high perplexity for their bigram then it is the place of separation.