from logreg_model import NGramsClassifier
import time
import configparser
from bert_model import TransformerLanguageDetection
from dataset import WILI2018Dataset


def main():

    start_time = time.time()
    config = configparser.ConfigParser()
    config.read('scripts/config.ini')
    if config['PIPELINE']['mode'] == 'LOGREG':
        classifier = NGramsClassifier()
        classifier.train()
        predictions = classifier.predict()
        f1_score = classifier.f1_score(predictions)
        print("f1 score on test dataset", f1_score)
    elif config['PIPELINE']['mode'] == 'BERT':
        train_set = WILI2018Dataset("train")
        test_set = WILI2018Dataset("test")
        classifier = TransformerLanguageDetection(device=config['BERT']['device'])
        classifier.train(train_set, batch_size=int(config['BERT']['batch_size']),
                         epochs=int(config['BERT']['epochs']))
        classifier.test(test_set, batch_size=int(config['BERT']['batch_size']))
        classifier.save_chekpoints()
    else:
        raise NotImplementedError

    classifier.save_chekpoints()
    print("checkpoint are saved to results folder")
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
