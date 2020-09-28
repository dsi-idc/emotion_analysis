import pandas as pd
import os
from transformers import AutoConfig, BertForSequenceClassification, AutoTokenizer
from collections import Counter
from transformers.data.processors import SingleSentenceClassificationProcessor
from transformers import Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, average_precision_score
import torch
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import pickle
import abc


class SentimentAnalyser(object):
    """ Class creating a running a sentiment analysis in Arabic
        Current class is a base class (kong of an abstract class) which only acts as a base line for other classes
        to inherit from

        Parameters
        ----------
        tokenizer : obj
            a tokenizer object to be used for tokanization procedures

        Attributes
        ----------
        results : dict
            dictionary containing the results
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.results = None

    @abc.abstractmethod
    def fit(self, train_df, dev_df):
        """an abstract function which will be implemented by other classes
        """
        pass

    @abc.abstractmethod
    def predict_proba(self, data):
        """an abstract function which will be implemented by other classes
        """
        pass

    @abc.abstractmethod
    def predict(self, data):
        """an abstract function which will be implemented by other classes
        """
        pass

    def eval_clf(self, y_true, y_pred, label_values=(0, 1), save_results=True):
        """
        a function to evaluate results of a sentiment model (classification)
        it allows an evaluation of a continuous prediction (probability)
        and of a categorical ones (final decision and not probability)

        Parameters
        ----------
        :param y_true: list/array
            list or array of the true values of the data. It must be in the same length as y_pred. It must contain
            only values given in label_values
        :param y_pred: list/array
            list or array of the predicted values of the data. It must be in the same length as y_pred. It can be either
            probability of categorical values (if categorical - must be same as the one given in label_values)
        :param label_values: tuple. defatul: (0,1)
            tuple with all values associated with the target column
        :param save_results: boolean. Default: True
            whether to save results or output is as part of the function's output

        :return: None or dict
            returns None is save_results=False
            returns a dictionary with results in case save_results=True
        """
        # converting the inputs to arrays anyway
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        n = len(y_true)
        if len(y_pred) != n:
            raise IOError("Input is invalid, length of both vectors is inconsistent")
        # if all true values are not from the given label values - problem
        if sum(np.logical_or(y_true == label_values[0], y_true == label_values[1])) != n:
            raise IOError("y_true values are inconsistent with the label_values given")
        # in case binary_preds==True, it means we were given binary values as input
        binary_preds = sum(np.logical_or(y_pred==label_values[0], y_pred==label_values[1])) == n
        # in case continous_preds==True, it means we were given [0,1] values of the prediction
        continous_preds = sum(np.logical_and(y_pred >= 0, y_pred <= 1)) == n and not binary_preds
        if not binary_preds and not continous_preds:
            raise IOError("y_pred values are inconsistent. They must contain probability values are label values")
        # actual calculations
        results = dict()
        if binary_preds:
            results['accuracy'] = accuracy_score(y_true, y_pred)
            results['f1_score'] = f1_score(y_true, y_pred)
            results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
            results['precision_score'] = precision_score(y_true, y_pred)
            results['recall_score'] = recall_score(y_true, y_pred)
        elif continous_preds:
            results['auc'] = roc_auc_score(y_true, y_pred)
            results['average_precision_score'] = average_precision_score(y_true, y_pred)

        # saving results or returning them as a dict
        if save_results:
            self.results = results
            return 0
        else:
            return results


class BOWBasedSentimentAnalyser(SentimentAnalyser):
    """ Class for running a bag-of-words sentiment analysis in arabic
        it inherits from the base class of SentimentAnalyser

        Parameters
        ----------
        saving_data_folder : str
            location of the folder to save the data generated by the model in (along the process we save generated data)

        saving_model_folder: str
            location of the folder to save results in

        bow_model_params: dict
            dictionary containing the parameters for the bag-of-words model

        tokenizer: obj. default: None
            a tokenizer object to be used for tokanization procedures

        label_col_name: str. deault: 'label'
            the name of the label column

        Attributes
        ----------
        pipeline : pipline object
            dictionary containing the results
        feature_names: list
            names of the explanatory features used by the model
        results: dict
            dictionary containing the results
    """
    def __init__(self, saving_data_folder, saving_model_folder, bow_model_params, tokenizer=None,
                 label_col_name='label'):
        super().__init__(tokenizer=tokenizer)
        self.saving_data_folder = saving_data_folder
        self.saving_model_folder = saving_model_folder
        self.bow_model_params = bow_model_params
        self.label_col_name = label_col_name
        # case data folder doens't exist - we'll create one
        if not os.path.exists(self.saving_data_folder):
            os.mkdir(self.saving_data_folder)
        self.pipeline = None
        self.feature_names = None
        self.results = None

    def fit(self, train_df, dev_df):
        """
        fitting the model based on the train set. validation is done using the dev set

        Parameters
        ----------
        :param train_df: dataframe
            a pandas dataframe containing data to be trained on

        :param dev_df: dataframe
            a pandas dataframe containing data to validate on

        :return: None
            all relevant results are saved under the object of the class. Next a prediction can be done
        """
        # setting up parameters for the model, in case it wasn't given as input in the dictionary
        min_df = 0.0001 if 'min_df' not in self.bow_model_params else self.bow_model_params['min_df']
        max_df = 0.95 if 'max_df' not in self.bow_model_params else self.bow_model_params['max_df']
        ngram_range = (1, 2) if 'ngram_range' not in self.bow_model_params else eval(self.bow_model_params['ngram_range'])
        classifier = RandomForestClassifier if 'clf' not in self.bow_model_params else eval(self.bow_model_params['clf'])
        # creating a pipeline for modeling
        pipeline = Pipeline([
            ('vect', TfidfVectorizer(min_df=min_df, max_df=max_df,
                                     analyzer='word', lowercase=False,
                                     ngram_range=ngram_range)),
            ('clf', classifier()),
        ])
        pipeline.fit(train_df['text'], train_df[self.label_col_name])
        self.pipeline = pipeline
        feature_names = pipeline.named_steps['vect'].get_feature_names()
        self.feature_names = feature_names

    def predict(self, test_df):
        """
        prediction new data, based on the model trained. Output here is categorical
        (for probability, use the  predict_proba)

        Parameters
        ----------
        :param test_df: dataframe
            the data to provide the prediction on

        :return: array
            an array with categorical predictions
        """
        prediction = self.pipeline.predict(test_df)
        return prediction

    def predict_proba(self, test_df):
        """
        prediction new data, based on the model trained. Output here is the probability to be assigned to each class

        Parameters
        ----------
        :param test_df: dataframe
            the data to provide the prediction on

        :return: array
            an array with probability predictions. Size of the array depends on the number of categories exist in the
            train label
        """
        try:
            prediction = self.pipeline.predict_proba(test_df)
        except AttributeError:
            print(f"Model of type {self.bow_model_params['clf']} doesn't have a predict_proba option, try using the "
                  f"simple 'predict' function.")
            raise AttributeError
        return prediction

    def save_model(self, folder_name, file_name):
        folder_to_save_into = os.path.join(self.saving_model_folder, folder_name)
        if not os.path.isdir(folder_to_save_into):
            os.system('mkdir -p ' + folder_to_save_into)
        pickle.dump(self.pipeline, open(os.path.join(folder_to_save_into, file_name + '.p'), "wb"))
        print(f"Model has been saved to {os.path.join(folder_to_save_into, file_name + '.p')}")
        return

    @staticmethod
    def load_model(folder_name, file_name):
        """
        a function to load the DL model from the disk
        :param folder_name: str
            folder location where the model has been saved
        :return: model (arabert)
            the model loaded
        """
        loaded_model = pickle.load(open(os.path.join(folder_name, file_name), "rb"))
        return loaded_model


class BertBasedSentimentAnalyser(SentimentAnalyser):
    """ Class for running a BERT sentiment analysis in arabic (using arabert)
        it inherits from the base class of SentimentAnalyser

        Parameters
        ----------
        saving_data_folder : str
            location of the folder to save the data generated by the model in (along the process we save generated data)

        saving_model_folder: str
            location of the folder to save results in

        bert_model_params: dict
            dictionary containing the parameters for the BERT model

        tokenizer: obj. default: None
            a tokenizer object to be used for tokanization procedures

        model_name: str. default: 'aubmindlab/bert-base-arabert'
            name of the BERT model to be used. It has to be recognized by the transformers package

        label_col_name: str. default: 'label'
            the name of the label column

        max_length: int. default: 128
            the maximum number of chars to take into account when an input as given to the BERT model

        Attributes
        ----------

        results: dict
            dictionary containing the results
    """
    def __init__(self, saving_data_folder, saving_model_folder, bert_model_params, tokenizer=None,
                 model_name='aubmindlab/bert-base-arabert', label_col_name='label', max_length=128):
        super().__init__(tokenizer=tokenizer)
        self.saving_data_folder = saving_data_folder
        self.saving_model_folder = saving_model_folder
        self.bert_model_params = bert_model_params
        self.model_name = model_name
        self.label_col_name = label_col_name
        # case data folder doens't exist - we'll create one
        if not os.path.exists(self.saving_data_folder):
            os.mkdir(self.saving_data_folder)
        self.max_length = max_length
        self.task_name = 'classification'
        self.results = None

    @staticmethod
    def compute_metrics(p):  # p should be of type EvalPrediction
        """
        an inner function for calculating few metrics along building the DL model
        taken from: https://github.com/huggingface/transformers/blob/master/src/transformers/trainer_utils.py
        :param p: obj (of type EvalPrediction)
            an EvalPrediction object type
        :return: dict
            a dictionary with results (f1 and friends)
        """
        preds = np.argmax(p.predictions, axis=1)
        assert len(preds) == len(p.label_ids)
        print(classification_report(p.label_ids, preds))
        print(confusion_matrix(p.label_ids, preds))
        f1_Positive = f1_score(p.label_ids, preds, pos_label=1, average='binary')
        f1_Negative = f1_score(p.label_ids, preds, pos_label=0, average='binary')
        macro_f1 = f1_score(p.label_ids, preds, average='macro')
        macro_precision = precision_score(p.label_ids, preds, average='macro')
        macro_recall = recall_score(p.label_ids, preds, average='macro')
        acc = accuracy_score(p.label_ids, preds)
        return {
            'f1_pos': f1_Positive,
            'f1_neg': f1_Negative,
            'macro_f1': macro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'accuracy': acc
        }

    def fit(self, train_df, dev_df):
        """
        fitting the model based on the train set. validation is done using the dev set

        Parameters
        ----------
        :param train_df: dataframe
            a pandas dataframe containing data to be trained on

        :param dev_df: dataframe
            a pandas dataframe containing data to validate on

        :return: None
            all relevant results are saved under the the location provided to save the model in.
            Next a prediction can be done
        """
        train_labels = Counter(train_df[self.label_col_name]).keys()
        num_labels = len(train_labels)
        dev_labels = Counter(train_df[self.label_col_name]).keys()
        if num_labels != len(dev_labels):
            raise IOError("train and dev datasets contain different number of labels")
        # creating a DF for train/test with relevant columns.
        # Not clear why the 'alpha' column is needed, but as written here
        # (https://towardsdatascience.com/https-medium-com-chaturangarajapakshe-text-classification-with-transformer-models-d370944b50ca) - it is required
        train_df = pd.DataFrame({
            'id': range(len(train_df)),
            'label': train_df[self.label_col_name],
            'alpha': ['a'] * train_df.shape[0],
            'text': train_df["text"].replace(r'\n', ' ', regex=True)
        })

        dev_df = pd.DataFrame({
            'id': range(len(dev_df)),
            'label': dev_df[self.label_col_name],
            'alpha': ['a'] * dev_df.shape[0],
            'text': dev_df["text"].replace(r'\n', ' ', regex=True)
        })
        # saving the DF to the new/old folder
        train_df.to_csv(os.path.join(self.saving_data_folder, "train.tsv"),
                        index=False, columns=train_df.columns, sep='\t', header=False)
        dev_df.to_csv(os.path.join(self.saving_data_folder, "dev.tsv"),
                      index=False, columns=dev_df.columns, sep='\t', header=False)

        config = AutoConfig.from_pretrained(self.model_name, num_labels=num_labels,
                                            output_attentions=True)  ##needed for the visualizations
        # loading the actual model to memory
        model = BertForSequenceClassification.from_pretrained(self.model_name, config=config)

        # Now we need to convert the examples in the dataset to features that the model can understand
        # this is a ready made class, provided by HuggingFace
        train_dataset = SingleSentenceClassificationProcessor(mode='classification')
        dev_dataset = SingleSentenceClassificationProcessor(mode='classification')

        # now adding examples (from the DF we created earlier) to the objects we created in the cell above)
        _ = train_dataset.add_examples(texts_or_text_and_labels=train_df['text'], labels=train_df[self.label_col_name],
                                       overwrite_examples=True)
        _ = dev_dataset.add_examples(texts_or_text_and_labels=dev_df['text'], labels=dev_df[self.label_col_name],
                                     overwrite_examples=True)

        train_features = train_dataset.get_features(tokenizer=self.tokenizer, max_length=self.max_length)
        test_features = dev_dataset.get_features(tokenizer=self.tokenizer, max_length=self.max_length)
        training_args = TrainingArguments("./train")

        training_args.do_train = True
        # setting the params of the BERT classifier
        for cur_param in self.bert_model_params.keys():
            try:
                training_args.__dict__[cur_param] = eval(self.bert_model_params[cur_param])
            except TypeError:
                training_args.__dict__[cur_param] = self.bert_model_params[cur_param]
        training_args.logging_steps = (len(train_features) - 1) // training_args.per_gpu_train_batch_size + 1
        training_args.save_steps = training_args.logging_steps
        training_args.output_dir = self.saving_model_folder
        training_args.eval_steps = 100
        # training_args.logging_dir = "gs://" from torch.utils.tensorboard import SummaryWriter supports google cloud storage

        trainer = Trainer(model=model,
                          args=training_args,
                          train_dataset=train_features,
                          eval_dataset=test_features,
                          compute_metrics=self.compute_metrics)
        trainer.train()
        # saving the model
        self.save_model(model=trainer.model, folder_name='bert_based_model')

    def predict(self, test_df):
        """
        prediction new data, based on the model trained. Output here is categorical
        (for probability, use the  predict_proba)

        Parameters
        ----------
        :param test_df: dataframe
            the data to provide the prediction on

        :return: array
            an array with categorical predictions
        """
        proba_prediciton = self.predict_proba(test_df=test_df)
        prediction = np.argmax(proba_prediciton, axis=1)
        return prediction

    def predict_proba(self, test_df):
        """
        prediction new data, based on the model trained. Output here is the probability to be assigned to each class

        Parameters
        ----------
        :param test_df: dataframe
            the data to provide the prediction on

        :return: array
            an array with probability predictions. Size of the array depends on the number of categories exist in the
            train label
        """
        # loading the model
        loaded_model = self.load_model(folder_name='bert_based_model')
        loaded_model.eval()
        # generating data for the prediction
        if type(test_df) == pd.Series:
            test_df = pd.DataFrame({
                'id': range(len(test_df)),
                'label': None,
                'alpha': ['a'] * test_df.shape[0],
                'text': test_df.replace(r'\n', ' ', regex=True)
            })
        else:
            test_df = pd.DataFrame({
                'id': range(len(test_df)),
                'label': test_df[self.label_col_name] if self.label_col_name in test_df.columns else None,
                'alpha': ['a'] * test_df.shape[0],
                'text': test_df["text"].replace(r'\n', ' ', regex=True)
            })
        test_dataset = SingleSentenceClassificationProcessor(mode='classification')
        _ = test_dataset.add_examples(texts_or_text_and_labels=test_df['text'], labels=test_df[self.label_col_name],
                                      overwrite_examples=True)
        test_features = test_dataset.get_features(tokenizer=self.tokenizer, max_length=self.max_length)
        test_features_input_ids = [i.input_ids for i in test_features]
        model_inputs_as_tensors = torch.tensor(test_features_input_ids)
        model_output = loaded_model(model_inputs_as_tensors)
        softmax_func = torch.nn.Softmax(dim=1)
        proba_pred = softmax_func(model_output[0])
        return proba_pred.detach().numpy()

    def save_model(self, model, folder_name):
        """
        a function to save the DL model in the disk
        :param model: model object
            the model (arabert) trained
        :param folder_name: str
            location where to save the model in
        """
        model.save_pretrained(os.path.join(self.saving_model_folder, folder_name))
        print(f"Model has been saved to {os.path.join(self.saving_model_folder, folder_name)}")
        return

    def load_model(self, folder_name):
        """
        a function to load the DL model from the disk
        :param folder_name: str
            folder location where the model has been saved
        :return: model (arabert)
            the model loaded
        """
        loaded_model = BertForSequenceClassification.from_pretrained(os.path.join(self.saving_model_folder, folder_name))
        return loaded_model
