from arabic_sentiment.data_loader import DataLoader
#from arabert.preprocess_arabert import preprocess
from arabert.preprocess import ArabertPreprocessor
from arabic_data_preprocess.pre_process import PreProcess
from arabic_sentiment.sentiment_clf import BertBasedSentimentAnalyser, BOWBasedSentimentAnalyser
import commentjson
import os
from transformers import AutoTokenizer
import pandas as pd
from transformers import AutoConfig, BertForSequenceClassification
from farasa.segmenter import FarasaSegmenter
# import logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

machine = 'azure_server'#'AVRAHAMI-PC'
if __name__ == "__main__":
    # loading the config file
    config_dict = commentjson.load(open('config.json'))
    if config_dict['mode'] != 'predict':
        raise IOError("Your are running a script for training a mode (or fine tune it), while the configuration in "
                      "the config.json file is set to something else. Either change the 'mode' to 'train' or run"
                      "another main file (probably 'main_sentiment_analysis_prediction.py'")
    # this is good when we have 2 classes
    # label_map = {'neg': 0, 'pos': 1}
    # this is good when we have 3 classes
    label_map = {'neg': 0, 'neu': 1, 'pos': 2, }
    label_col = 'label'
    # loading the data
    data_loader_obj = DataLoader(files_path=config_dict['data_path'][machine], target_column_idx=0)
    dev_tsvs = [f_name for f_name in os.listdir(config_dict['data_path'][machine])
                if not f_name.startswith('train') and f_name.endswith('.tsv')]
    dev_sentiment_df = \
        data_loader_obj.load_tsv_files(file_names=dev_tsvs, new_column_names=[label_col, 'text'],
                                       seed=config_dict['random_seed'], sample_rate=config_dict['sample_rate'])
    # preprocessing the data to be used by the clf later. If it is a BOW model - we'll pull out and paste the emojies
    preprocess_obj = PreProcess(use_default_farsa_preprocess=True)

    #dev_sentiment_df = pd.read_csv(
    #    'C:\\Users\\avrahami\\Documents\\Private\\IDC\\influencer_influenced_project\\sentiment_analysis_effort\\unlabeled_tweets_for_yotam.csv')
    dev_sentiment_df['text'] =\
        preprocess_obj.transform(sentences_list=dev_sentiment_df['text'],
                                 extract_and_paste_emojies=True if config_dict['model_type'] == 'BOW' else False)
    dev_sentiment_df[label_col] = dev_sentiment_df[label_col].apply(lambda x: label_map[x])
    saving_models_path = os.path.join(config_dict['saving_models_path'][machine], config_dict['model_version'])
    if config_dict['model_type'] == 'arabert' or config_dict['model_type'] == 'gigabert':
        dev_label = dev_sentiment_df[label_col]
        bert_model_name = config_dict['bert_model_name']
        # fitting the model
        tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        bert_params_dict = config_dict['bert_model_params']
        bert_params_dict['seed'] = config_dict['random_seed']
        bert_based_sentiment_obj = \
            BertBasedSentimentAnalyser(saving_data_folder=config_dict['saving_data_path'][machine],
                                       saving_model_folder=saving_models_path,
                                       bert_model_params=bert_params_dict, tokenizer=tokenizer,
                                       model_name=bert_model_name, label_col_name='label', max_length=128)

        # probability prediction and evaluation (possible and relevant mainly for binary classification)
        if len(label_map) == 2:
            dev_set_predictions = bert_based_sentiment_obj.predict_proba(test_df=dev_sentiment_df['text'])
            dev_positive_prediction = dev_set_predictions[:, 1]
            bert_based_sentiment_obj.eval_clf(y_true=dev_label, y_pred=dev_set_predictions[:, 1], save_results=True)
            print(bert_based_sentiment_obj.results)

        # categorical prediction and evaluation (possible and relevant mainly for multi classification problems)
        elif len(label_map) > 2:
            dev_set_predictions = bert_based_sentiment_obj.predict(test_df=dev_sentiment_df)
            bert_based_sentiment_obj.eval_clf(y_true=dev_label, y_pred=dev_set_predictions, label_values=(0, 1, 2), save_results=True)
            print(bert_based_sentiment_obj.results)

    elif config_dict['model_type'] == 'BOW':
        dev_label = dev_sentiment_df[label_col]
        bow_based_sentiment_obj = BOWBasedSentimentAnalyser(saving_data_folder=config_dict['saving_data_path'][machine],
                                                            saving_model_folder=saving_models_path,
                                                            bow_model_params=config_dict['bow_model_params'])
        bow_trained_model = \
            BOWBasedSentimentAnalyser.load_model(os.path.join(config_dict['saving_models_path'][machine],
                                                              config_dict['model_version']), file_name='bow_model')
        bow_based_sentiment_obj.pipeline = bow_trained_model
        # in case we have a model which can run the 'predict_proba' function
        dev_set_predictions = bow_based_sentiment_obj.predict_proba(test_df=dev_sentiment_df['text'])
        dev_positive_prediction = dev_set_predictions[:, 1]

        bow_based_sentiment_obj.eval_clf(y_true=dev_label, y_pred=dev_positive_prediction, save_results=True)
        print("Here are the results of the model over the dev dataset:")
        print(bow_based_sentiment_obj.results)

        #saving the model created into a folder
        #bow_based_sentiment_obj.save_model(folder_name='bow_based_models', file_name='bow_model_28.9.2020')




