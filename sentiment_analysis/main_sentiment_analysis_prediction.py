from arabic_sentiment.data_loader import DataLoader
#from arabert.preprocess_arabert import preprocess
from arabert.preprocess import ArabertPreprocessor
from arabic_data_preprocess.pre_process import PreProcess
from arabic_sentiment.sentiment_clf import BertBasedSentimentAnalyser, BOWBasedSentimentAnalyser
import commentjson
import os
from transformers import AutoTokenizer, AutoConfig, Trainer
import pandas as pd
from transformers.data.processors import SingleSentenceClassificationProcessor
from transformers import TrainingArguments, BertForSequenceClassification
from farasa.segmenter import FarasaSegmenter
import torch
from scipy.special import softmax
import numpy as np
import copy
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
    label_map = {'NEG': 0, 'NEU': 1, 'POS': 2}
    label_col = 'label'
    # loading the data
    data_loader_obj = DataLoader(files_path=config_dict['data_path'][machine], target_column_idx=0)
    dev_tsvs = [f_name for f_name in os.listdir(config_dict['data_path'][machine])
                if f_name.startswith('test_data_challange') and f_name.endswith('.tsv')]
    dev_sentiment_df = \
        data_loader_obj.load_tsv_files(file_names=dev_tsvs, new_column_names=[label_col, 'text'],
                                       seed=config_dict['random_seed'], sample_rate=config_dict['sample_rate'])
    orig_dev_sentiment_df = copy.deepcopy(dev_sentiment_df)
    # preprocessing the data to be used by the clf later. If it is a BOW model - we'll pull out and paste the emojies
    preprocess_obj = PreProcess(use_arabert_preprocess=True, tokenizer=None, lemmetize_data=False)

    dev_sentiment_df['text'] = \
        preprocess_obj.transform(sentences_list=dev_sentiment_df['text'], keep_emojis=True,
                                 remove_RT_prefix=True, remove_punctuations=True)
    dev_sentiment_df[label_col] = dev_sentiment_df[label_col].apply(lambda x: label_map[x])
    saving_models_path = os.path.join(config_dict['saving_models_path'][machine], config_dict['model_version'])
    if config_dict['model_type'] == 'arabert' or config_dict['model_type'] == 'gigabert' or config_dict['model_type'] =='marbert':
        dev_label = dev_sentiment_df[label_col]
        bert_model_name = config_dict['bert_model_name']

        ## NEW CODE (19.1.2021)
        config = AutoConfig.from_pretrained(config_dict['bert_model_name'], num_labels=len(label_map),
                                            output_attentions=True)
        loaded_model = BertForSequenceClassification.from_pretrained(os.path.join(config_dict['saving_models_path'][machine],
                                                                                  config_dict['model_version']))
        dev_df = pd.DataFrame({
            'id': range(len(dev_sentiment_df)),
            'label': dev_sentiment_df[label_col],
            'alpha': ['a'] * dev_sentiment_df.shape[0],
            'text': dev_sentiment_df["text"].replace(r'\n', ' ', regex=True)
        })
        dev_df.to_csv(os.path.join(config_dict['saving_models_path'][machine], "dev.tsv"),
                      index=False, columns=dev_df.columns, sep='\t', header=False)
        dev_dataset = SingleSentenceClassificationProcessor(mode='classification')
        _ = dev_dataset.add_examples(texts_or_text_and_labels=dev_sentiment_df['text'],
                                     labels=dev_sentiment_df[label_col], overwrite_examples=True)
        tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        dev_features = dev_dataset.get_features(tokenizer=tokenizer, max_length=128)

        class MyTrainer(Trainer):
            def __init__(self, loss_func=torch.nn.CrossEntropyLoss(), **kwargs):
                self.loss_func = loss_func
                super().__init__(**kwargs)

            def compute_loss(self, model, inputs):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs[0]
                return self.loss_func(logits, labels)


        args = TrainingArguments(
            "arabic_nlp_model",
            evaluation_strategy="epoch",
            learning_rate=1e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=8,
            num_train_epochs=5,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1_PN",
        )

        # setting the params of the BERT classifier
        bert_model_params = config_dict['bert_model_params']
        bert_model_params['seed'] = config_dict['random_seed']
        for cur_param in bert_model_params.keys():
            try:
                args.__dict__[cur_param] = eval(bert_model_params[cur_param])
            except TypeError:
                args.__dict__[cur_param] = bert_model_params[cur_param]
        args.save_steps = args.logging_steps
        trainer = MyTrainer(model=loaded_model, args=args, train_dataset=dev_features,
                            eval_dataset=dev_features,
                            compute_metrics=BertBasedSentimentAnalyser.compute_metrics)
        # splitting the prediction set to bulks of 1000 so we will not meet memory errors)
        pred_bulks = 1001
        all_predictions = list()
        for cur_idx in range(int(len(dev_features)/pred_bulks) + 1):
            cur_bulk = dev_features[pred_bulks*cur_idx:pred_bulks*(cur_idx+1)]
            trainer_predictions = trainer.predict(cur_bulk)
            all_predictions.append(trainer_predictions.predictions[0])
        # creating the predictions (proba)
        all_predictions_array = np.concatenate(all_predictions, axis=0)
        proba_predictions = np.apply_along_axis(softmax, 1, all_predictions_array)
        proba_predictions_df = pd.DataFrame(proba_predictions, columns=['neg_proba', 'neu_proba', 'pos_proba'])
        dev_df_with_predictions = pd.concat([orig_dev_sentiment_df, proba_predictions_df], axis=1)
        dev_df_with_predictions.to_csv("test_data_challange_sentiment_predictions_with_label.tsv", index=False, sep='\t')
        print(trainer_predictions.metrics)
        # OLD CODE (without usage of the trainer)
        """
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
        """
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




