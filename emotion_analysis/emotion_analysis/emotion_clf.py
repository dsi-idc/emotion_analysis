# Model version: 0.1
# Description: arabic emotion analysis project - 26.9.2020
# Contributor: Yotam Nahum
import pandas as pd
from nltk.tokenize import WordPunctTokenizer

punct_tokenizer = WordPunctTokenizer()
# from transformers import AutoTokenizer, AutoModel
# from arabert.preprocess_arabert import never_split_tokens, preprocess
from simpletransformers.classification import MultiLabelClassificationModel
from sklearn.metrics import classification_report
from arabic_data_preprocess.pre_process import PreProcess
import commentjson
#from utils.utils import blockPrint, enablePrint


class EmotionMultilabelClassification:

    def __init__(self,
                 emotion_list: list = None,
                 emotion_weight: list = None,
                 use_cuda: bool = False,
                 config_file: str = './config.json',
                 verbose=True,
                 model_args: dict = None):

        """
        **Build a BERT model based on the pre-trained BERT model as specified on the "config.json" file under
         'arabert_model_name' parameter.**

        parameters:
        ----------
            :param emotion_list: type: list. default: None. List of emotion names. If None all the emotion weights will
            be set as specified on the "config.json" file under 'emotion_list' parameter.
            :param emotion_weight: type: list. default: None. The emotion weight by label. If None all the emotion weights
            will be set to 1.
            :param use_cuda: type: bool. default: False. If True, will use GPU.
            :param config_file: type: str. Default: "./config.json". The config file location.
            :param verbose: type: bool. default: True. Enable verbose.
            :param model_args: type: dict. default: None. Model arguments. If None, the arguments will be imported from the
                "config_file" file.
                reprocess_input_data: default True. If True, the input data will be
                    reprocessed even if a cached file of the input data exists in
                    the cache_dir.
                'overwrite_output_dir': default True. If True, the trained model will
                    be saved to the output_dir and will overwrite existing saved
                    models in the same directory.
                'num_train_epochs': default 1. The number of epochs the model will be
                    trained for.
                'train_batch_size': default 8. The training batch size.
                'learning_rate': default 4e-5. The learning rate for training.
                'evaluate_during_training': default True. Set to True to perform evaluation
                    while training models. Make sure eval data is passed to the training
                    method if enabled.
                'evaluate_during_training_steps': default 380. Perform evaluation at every
                    specified number of steps. A checkpoint model and the evaluation
                    results will be saved.
                'save_model_every_epoch': default False. Save a model checkpoint at the
                    end of every epoch.
                'save_eval_checkpoints': default False. Save a model checkpoint for every
                    evaluation performed.
                'evaluate_during_training_verbose': default True. Print results from
                    evaluation during training.
                'use_cached_eval_features': default True. Evaluation during training uses
                    cached features. Setting this to False will cause features to be
                    recomputed at every evaluation step.
                'save_optimizer_and_scheduler': default False. Save optimizer and scheduler
                    whenever they are available.
                'output_dir': default “outputs/”. The directory where all outputs will be stored.
                    This includes model checkpoints and evaluation results.
                'fp16': default True. Whether or not fp16 mode should be used. Requires Nvidia
                    Apex library.
                'threshold': default 0.5. The threshold for the emotion probability to set the specific emotion
                    as True (relevant to the 'predict' function.
        attributes:
        ----------
            evaluation_report:
                Init to None
                type: str.
                Evaluation report of the model performance with sklearn.metrics.classification_report.
            """

        config_dict = commentjson.load(open(config_file))
        if emotion_list is None:
            self.emotion_list = config_dict['emotion_list']
        self.num_of_features = len(self.emotion_list)
        if emotion_weight is None:
            self.emotion_weight = [1 for x in range(self.num_of_features)]
        if model_args is None:
            self.model_args = config_dict['bert_model_params']
        self.use_cuda = use_cuda
        self.verbose = verbose
        self.model = MultiLabelClassificationModel('bert',
                                                   config_dict['arabert_model_name'],
                                                   num_labels=self.num_of_features,
                                                   pos_weight=self.emotion_weight,
                                                   use_cuda=use_cuda,
                                                   args=self.model_args)
        self.evaluation_report = None
        # return self.model

    def train(self, train_data: pd.DataFrame, eval_data: pd.DataFrame):
        """
        **Train the model.**

        parameters:
        ----------
            :param train_data:
                type: pd.DataFrame.
                A train DataFrame with two columns:
                "text" column (containing the input text after farasa_tokenization).
                "labels" column (list type object of binary labels).
            :param eval_data:
                type: pd.DataFrame.
                An eval DataFrame with two columns:
                "text" column (containing the input text after farasa_tokenization).
                "labels" column (list type object of binary labels).

        returns:
        --------
            :return: type: object. Trained BERT model. The trained model will be saved locally in the output_dir folder as
                set in the config file ("outputs\" as Default).
        """
        model = self.model
        trained_model = model.train_model(train_data, eval_df=eval_data)
        return trained_model

    def predict(self,
                input_data: pd.DataFrame,
                pretrained_model: str = None,
                return_df: bool = False,
                results_to_csv: bool = False):
        """
        **Predict emotions from given text.**

        parameters:
        ----------
            :param input_data:
                type: pd.DataFrame.
                A Pandas DataFrame with minimum one column.
                The DataFrame must have a column named "text" which contains the input text after farasa_tokenization.
            :param pretrained_model:
                type: str.
                default: None.
                Location of a pretrained BERT model. If None, will use the value from the config
                file ("outputs\" as Default).
            :param return_df:
                type: bool.
                default: False.
                If set to True return Pandas dataframe with prediction and probabilities. If set to False return an
                array of the raw probabilities. Default False.
            :param results_to_csv:
                type: bool.
                default: False.
                If set to True saves a csv file with prediction and probabilities in the output_dir folder as
                set in the config file ("outputs\" as Default). Default False.

        returns:
        --------
            :return:
            If return_df==True:
                Pandas dataframe with prediction and probabilities.
                type: pd.Dataframe.
            If return_df==False:
                type: np.array.
                Array of the raw probabilities.
                Default False.
        """

        if pretrained_model is None:
            pretrained_model = self.model_args['output_dir']
        self.model = MultiLabelClassificationModel('bert', pretrained_model, num_labels=self.num_of_features,
                                                   use_cuda=self.use_cuda)
        predictions, raw_outputs = self.model.predict(input_data['text'])
        predict_df = pd.DataFrame(raw_outputs, columns=self.emotion_list)
        predict_df = predict_df.add_suffix('_raw')
        pred = pd.DataFrame(predictions, columns=self.emotion_list)
        predict_df['predictions'] = pred.dot(pd.Index(self.emotion_list) + ', ').str.strip(', ')
        predict_df['text'] = input_data['text']

        if results_to_csv is True:
            predict_df.to_csv(self.model_args['output_dir']+'predictions.csv')
        if return_df is True:
            return predict_df
        else:
            return raw_outputs

    def eval(self, data: pd.DataFrame, pretrained_model: str = None):
        """
        **Evaluate the model performance with classification_report from sklearn.
         (for multi-label classification).**

        parameters:
        ----------
            :param data: A Pandas DataFrame with minimum two columns.
                type: pd.DataFrame.
                "text" column (containing the input text after farasa_tokenization).
                "labels" column (list type object of binary labels) - the ground truth.
            :param pretrained_model:
                type: str.
                Location of a pretrained BERT model. If None, will use the value from the config
                file ("outputs\" as Default).

        returns:
        --------
            :return: (predict_df,performance_report)
            type: A tuple of a dataframe and a string:

                predict_df: A Pandas DataFrame with 3 columns:
                    "text" column : containing the input text after farasa_tokenization.
                    "ground_truth" column : the ground truth labels (multi-label).
                    "predictions" column : the predicted labels (multi-label).
                evaluation_report:
                    type: str.
                    Evaluation report of the model performance with sklearn.metrics.classification_report.
        """
        if pretrained_model is None:
            pretrained_model = self.model_args['output_dir']

        self.model = MultiLabelClassificationModel('bert', pretrained_model,
                                                   num_labels=self.num_of_features,
                                                   use_cuda=self.use_cuda)

        predictions, raw_outputs = self.model.predict(data['text'])
        ground_truth = data['labels'].to_list()
        evaluation_report = classification_report(ground_truth,
                                                  predictions,
                                                  target_names=self.emotion_list,
                                                  zero_division=0)
        if self.verbose:
            print(evaluation_report)
        predict_df = data.copy()
        pred = pd.DataFrame(predictions, columns=self.emotion_list)
        predict_df['predictions'] = pred.dot(pd.Index(self.emotion_list) + ', ').str.strip(', ')
        gt = pd.DataFrame(ground_truth, columns=self.emotion_list)
        predict_df['ground_truth'] = gt.dot(pd.Index(self.emotion_list) + ', ').str.strip(', ')

        return predict_df[['text', 'ground_truth', 'predictions']], evaluation_report

    def fit(self, train_data: pd.DataFrame, eval_data: pd.DataFrame):
        """
        **Preprocess the data and train the model.** The model will be saved in the folder defined in the config file
        under 'output_dir' (“outputs/” as default). This function is a pipeline of 3 other functions:

            **pre_process.fit:** Fits the emotion_weight parameter to the label distribution (multilabel) on a given data
            set. The emotion_weight is taking into account in the model's training process.

            **pre_process.transform**: Transforms a given Pandas DataFrame into the correct input shape.

            **emotion_clf.train**: Train the model.

        parameters:
        ----------
            :param train_data:
                type: pd.DataFrame.
                A train Pandas DataFrame with two types of columns:
                    "text" column: containing the input text.
                    label columns: should be structured as dummies (binary column for each label)
            :param eval_data:
            type: pd.DataFrame.
                An evaluation Pandas DataFrame with two types of columns:
                    "text" column: containing the input text.
                    label columns: should be structured as dummies (binary column for each label)

        returns:
        --------
            :return: Trained BERT model. The trained model will be saved locally in the output_dir folder as
                    set on the "Build_model" function.

        Examples
        --------
        >>> import pandas as pd
        >>> from emotion_analysis.emotion_clf import EmotionMultilabelClassification
        # Load the datasets (you can import 'dataset/small.csv' instead for a quick first run)
        >>> train_data = pd.read_csv('dataset/train_data.csv')
        >>> eval_data = pd.read_csv('dataset/eval_data.csv')
        >>> test_data = pd.read_csv('dataset/test_data.csv')

        # Preprocess the data -> train the model
        >>> emc = EmotionMultilabelClassification() # Init the model
        >>> emc.fit(train_data,eval_data) # Train the model

        # Preprocess new data -> predict results
        >>> eval_df = emc.transform(test_data,return_df=True,evaluate=True) # Set 'evaluate=True' to get the model performance report
        >>> print (test_data) # Print the dataframe
        >>> print(emc.evaluation_report) # Print the model performance report
        """
        preprocess = PreProcess(emotion_list=self.emotion_list,verbose=self.verbose)
        preprocess.fit(train_data)
        train_df, train_df_full = preprocess.transform(train_data)
        eval_df, eval_df_full = preprocess.transform(eval_data)
        self.train(train_df, eval_df)

    def transform(self,
                  new_data: pd.Series,
                  return_df: bool = False,
                  results_to_csv: bool = False,
                  evaluate: bool = False,
                  pretrained_model: str = None):
        """
        **Preprocess new data and predict the emotions**. The model will be uploaded from the folder defined in the
        config file under 'output_dir' (“outputs/” as default).

        This function is a pipeline of 3 other functions:

            **pre_process.transform**: Transforms a given Pandas DataFrame into the correct input shape.

            **emotion_clf.predict**: Predict emotions from given text.

        parameters:
        ----------
            :param new_data:
            type: pd.DataFrame.
                A train Pandas DataFrame with two types of columns:
                    "text" column: containing the input text.
                    label columns: should be structured as dummies (binary column for each label)
            :param return_df:
                type: bool.
                default: False.
                If set to True return Pandas dataframe with prediction and probabilities. If set to False return an
                array of the raw probabilities.
            :param results_to_csv:
                type: bool.
                default: False.
                If True, saves a csv file with prediction and probabilities in the output_dir folder as
                set in the config file ("outputs\" as Default).
            :param evaluate:
                type: bool.
                default: False.
                If True, will evaluate the model results. To use this option, the input dataframe must include label
                columns.
            :param pretrained_model:
                type: str.
                default: None.
                Location of a pretrained BERT model. If None, will use the value from the config
                file ("outputs\" as Default).

        attributes:
        ----------
            evaluation_report:
                type: str.
                Evaluation report of the model performance with sklearn.metrics.classification_report.

        returns:
        --------
            :return:
                If return_df==True:
                    type: pd.Dataframe.
                    Pandas dataframe with prediction and probabilities.
                If return_df==False:
                    type: np.array.
                    Array of the raw probabilities.
                    Default False.

        Examples
        --------
        # Pipeline #1: preprocess new data -> init the model -> predict results
        >>> import pandas as pd
        >>> from emotion_analysis.emotion_clf import EmotionMultilabelClassification
        >>> unlabeled_data =["انها مثل هذا اليوم الجميل","كان لدي أسبوع مرهق ... لا بد لي من العودة إلى المنزل"]
        >>> emc = EmotionMultilabelClassification() # Init the model
        # Return a full dataframe
        # Save the full results (prediction and probabilities) to csv
        >>> prediction_df = emc.transform(unlabeled_data,return_df=True,results_to_csv=True)
        >>> print(prediction_df)
        # Return only the array of probabilities
        >>> probabilities = emc.transform(unlabeled_data)
        >>> print(probabilities)

        # Pipeline #2: preprocess evaluation data -> init the model -> evaluate results
        >>> import pandas as pd
        >>> from emotion_analysis.emotion_clf import EmotionMultilabelClassification
        >>> eval_data = pd.read_csv('dataset/eval_data.csv') # Load evaluation data
        >>> emc = EmotionMultilabelClassification(verbose=False) # Init the model, disable verbose (optional)
        >>> eval_df = emc.transform(eval_data,return_df=True,evaluate=True) # Set 'evaluate=True' to get the model performance report
        >>> print (eval_df) # Print the dataframe
        >>> print(emc.evaluation_report) # Print the model performance report
        """

        preprocess = PreProcess(emotion_list=self.emotion_list,verbose=self.verbose)

        if evaluate is True:
            input_data, input_data_full = preprocess.transform(new_data, labeled_data=True)
            predict_df,evaluation_report = self.eval(input_data, pretrained_model=pretrained_model)
            self.evaluation_report = evaluation_report
            if results_to_csv is True:
                predict_df.to_csv(self.model_args['output_dir'] + 'evaluate.csv')
        else:
            input_data, input_data_full = preprocess.transform(new_data, labeled_data=False)
            predict_df = self.predict(input_data, return_df=return_df, results_to_csv=results_to_csv,pretrained_model=pretrained_model)
        return predict_df
