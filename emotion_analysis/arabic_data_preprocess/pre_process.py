# Model version: 0.1
# Description: pre-preprocess module for arabic emotion analysis project - 26.9.2020
# Contributor: Yotam Nahum

import pandas as pd
from transformers import AutoTokenizer
from arabert.preprocess_arabert import preprocess, never_split_tokens
from farasa.segmenter import FarasaSegmenter
import preprocessor as p
import commentjson
from utils.utils import blockPrint, enablePrint


class PreProcess(object):
    def __init__(self,
                 use_default_farsa_preprocess: bool = True,
                 tokenizer: object = None,
                 lemmetize_data: bool = False,
                 emotion_list: list = None,
                 config_file: str = './config.json',
                 verbose=True):
        """
        **Initiate the class.**

        parameters:
        ----------
            :param use_default_farsa_preprocess:
                If True, Apply farasa_tokenization on the text
                    (see https://github.com/MagedSaeed/farasapy for more details). Default True.
            :param tokenizer:
                type: object. default: None. Custom tokenizer. if None will take the tokenizer from the
                    pretrained model.
            :param lemmetize_data:
                Not in use. type: bool. default: False.
            :param emotion_list:
                type: list. default: None. List of emotion names. If None all the emotion weights will be set as
                    specified on the "config.json" file under 'emotion_list' parameter.
            :param config_file:
                type: str. default: "./config.json". The config file location.
            :param verbose:
                type: bool. default: True. Enable verbose

        attributes:
        ----------
            emotion_weight: Sets the emotion_weight parameter to None. The parameter will be set through the "fit"
                function.
            num_of_features: The number of different emotions (different labels) in the dataset. The attribute
                is calculated by the len of the "emotion_list" length.
        """
        self.use_default_farsa_preprocess = use_default_farsa_preprocess
        config_dict = commentjson.load(open(config_file))
        # bert_model_name = config_dict['arabert_model_name']
        if tokenizer is None:
            # we will use the built model by aubmindlab (https://huggingface.co/aubmindlab/bert-base-arabert)
            model_name = config_dict['arabert_model_name']
            self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                           do_lower_case=False,
                                                           do_basic_tokenize=True,
                                                           never_split=never_split_tokens)
        else:
            self.tokenizer = tokenizer
        self.lemmetize_data = lemmetize_data

        if emotion_list is None:
            emotion_list = config_dict['emotion_list']

        self.emotion_list = emotion_list
        self.num_of_features = len(self.emotion_list)
        self.emotion_weight = None
        self.verbose = verbose

    def fit(self, train_data: pd.DataFrame) -> pd.DataFrame:
        """
        **Fits the emotion_weight parameter to the label distribution (multilabel) on a given data set.
        The emotion_weight is taking into account in the model's training process.**

        parameters:
        ----------
            :param train_data:
                Pandas DataFrame of the training input samples. The label columns should be structured
                    as dummies (binary column for each label).

        returns:
        --------
            :return emotion_weight:
                Pandas DataFrame of the emotion_weight by label.

        """
        relative = train_data[self.emotion_list].sum(axis=0, skipna=True).values
        emotion_weight = ((len(train_data) - relative) / relative).tolist()
        self.emotion_weight = emotion_weight
        if self.verbose:
            print(pd.DataFrame(self.emotion_weight, index=self.emotion_list))
        return emotion_weight

    def transform(self, data, dataset_type='multilabel', labeled_data=True):
        """
        **Transforms a given Pandas DataFrame into the correct input shape.**

            Remove MENTIONS,RESERVED and URL from the text (see https://pypi.org/project/tweet-preprocessor/
            for more details).

            Convert the multiple label columns into one binary column named "labels" (Series of list type
            objects).

            Apply farasa_tokenization on the text (see https://github.com/MagedSaeed/farasapy for more
            details).

        parameters:
        ----------
            :param data:
                type: str, list, np.array, pd.Series, or pd.Dataframe.
                An object that containing multiple text strings.
                if labeled_data parameter is set to True, the input must be a Pandas DataFrame with minimum two types
                 of columns:
                    data['text']: containing the original text.
                    data[['fear', 'joy', 'anger', etc.]] (optional): containing The labels. The labels columns
                        should be structured as dummies (binary column for each label).
            :param dataset_type:
                type: str.
                default: 'multilabel'.
                Set the preprocessing type of the target labels.'classification' or 'multilabel',
            :param labeled_data:
                type: bool.
                default: True.
                Need to set to True if the data contains a column of the target labels.

        returns:
        --------
            :return: (input_data,full_data)
            type: tuple.
            A tuple of two dataframes:
             input_data : Pandas DataFrame with two columns:
                        "text" column (containing the input text after farasa_tokenization).
                        "labels" column (list type object of binary labels). The "labels" column would not
                        be passed if "labeled_data" parameter set to False.
            full_data : Pandas DataFrame similar to "input_data", with the original columns
                        (for data exploration purposes).
        """
        #if not self.verbose:
            #blockPrint()
        if type(data) == pd.core.frame.DataFrame:
            if 'text' in data.columns:
                pass
            else:
                raise ValueError("The dataframe do not have a column named 'text'.")
        else:
            try:
                if type(data) == str:
                    data = [data]
                temp = pd.DataFrame(data,columns=['text'])
                data = temp.copy()
            except Exception:
                raise ValueError("The input data type is not supported. \n"
                                 " Please use data type of list, array, pd.Series or pd.Dataframe.")
        farasa_segmenter = FarasaSegmenter(interactive=True)
        if dataset_type == 'classification':
            new_sentences_list = list()
            for cur_text in data:
                # we must handle here the emojies!! it is currently removed due to the preprocessing
                if self.use_default_farsa_preprocess:
                    new_sentences_list.append(preprocess(cur_text,
                                                         do_farasa_tokenization=True,
                                                         farasa=farasa_segmenter,
                                                         use_farasapy=True))
                # currently not doing anything in such case, only supports the default case
                else:
                    new_sentences_list.append(cur_text)

            return new_sentences_list

        if dataset_type == 'multilabel':
            p.set_options(
                p.OPT.MENTION,
                # p.OPT.HASHTAG,
                p.OPT.RESERVED,
                # p.OPT.EMOJI,
                # p.OPT.SMILEY,
                p.OPT.URL)
            temp = data.copy()
            # temp['text'] = temp['Tweet']

            for x in temp.index:
                input_file_name = temp.loc[x,'text']
                til = p.clean(str(input_file_name))
                til = til.replace('\\n', ' ')
                til = til.replace('#', '')
                til = preprocess(til, do_farasa_tokenization=True, farasa=farasa_segmenter, use_farasapy=True)
                # til = arabert_tokenizer.tokenize(text_preprocessed)
                temp.loc[x,'text'] = til

            if labeled_data:
                tp = temp[self.emotion_list].values
                temp['labels'] = pd.Series(object)
                temp['labels'] = tp.tolist()
                temp.reset_index(inplace=True)
                full_data = temp
                input_data = full_data[['text', 'labels']]
            else:
                full_data = temp
                input_data = full_data[['text']]

            return input_data, full_data

        #if not self.verbose:
            #enablePrint()
