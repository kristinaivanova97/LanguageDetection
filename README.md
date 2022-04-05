### Language Detection

The project is to find languages in which the text was written.
The language list was shorten to just Russian, Ukrainian, Belarusian, Kazakh, Azerbaijani, Armenian, Georgian, Hebrew, 
English and German. \
One of the most common and simple approaches is Logistic Regression with TFIDF encodings, which was implemented here 
as a baseline. \
Nevertheless, transformers are known to be very good at language understanding, so as a second approach 
"bert-base-multilingual-cased" model was trained and compared to logreg one. \
Repository contents: 

##### _scripts folder_
- initializer.py - some preparations for future work like dataset downloading, some parameter initializations.
- metrics.py - class for metric computations (accuracy, f1).
- logreg_model.py - class with logreg model (with which you can train, test your model and make sentence predictions).
- dataset.py - class for dataset preparation for Bert model.
- bert_model.py - class with bert model (with which you can train, test your model and make sentence predictions).
- train.py - script to train your model (logreg or Bert), the training configurations and model type are set in config.ini file.
- predict.py - with it you can get languages predictions (one or many) for an input sentence (from terminal).
##### _data folder_
- essential dataset parts are held in data folder.
##### _results folder_
- resulting checkpoints are saved to results folder.

#### Dataset description

Some description you can see at dataset repo (https://github.com/huggingface/datasets/tree/master/datasets/wili_2018). \
installation: pip install dataset. \
Dataset contains sentence examples for 235 languages (500 samples for each language). It was formed from Wikipedia.

#### Results

As was mentioned above, I used 2 methods for detecting languages in the text. Both methods can predict one or many languages, 
depending on threshold. 

1) Char-level ngram logistic regression model with tfidf encodings. Showed high accuracy level (). 
It learns char ngrams (from one grams to five grams) from scratch, so it could be easily broadcast to other languages. 
Moreover, this approach showed good generalisation ability even on not so big datasets.

2) Bert with its own tokenizer, which splits words on meaningful (bpe) subparts. These model was a bit worse, 
but it could be easily explained, Bert is multilingual, so it knows much extra infomation. 
Moreover, we retrained only classification layer, encoding layers were freezed as our dataset and computer abilities are not enough for appropriate retraining of the whole Bert.
Nevertheless, accuracy is still high (0.98 on test set).