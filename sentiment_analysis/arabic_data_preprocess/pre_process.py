from transformers import AutoTokenizer
#from arabert.preprocess_arabert import preprocess, never_split_tokens
from arabert.preprocess import ArabertPreprocessor, never_split_tokens
from farasa.segmenter import FarasaSegmenter
import re
import emoji
import string


class PreProcess(object):
    """ Class for doing all pre-processing steps for any arabic NLP projects

        Parameters
        ----------

        Attributes
        ----------
        use_arabert_preprocess : boolean. Default: True
            whether to use the default farasa (a python package) for pre-processing procedures
        tokenizer : tokenizer object. Default: None
            the tokenizer to be used along the pre-processing steps. Currently it is not used - for future usage.
            In case None is given - the arabert (a python package) tokenizer is being used
        lemmetize_data : boolean. Default: False
            Whether to run lemmatization over the data. Currently it is not used - for future usage.

        Examples
        --------
        >>> list_of_sentences = ['هذه جملة رائعة', 'ياله من يوم جميل 🤔 🙈']
        >>> preprocess_obj = PreProcess(use_default_farsa_preprocess=True)
        >>> preprocess_obj.transform(sentences_list=list_of_sentences, extract_and_paste_emojies=True)
        """
    def __init__(self, use_arabert_preprocess=True, tokenizer=None, lemmetize_data=False):
        self.use_arabert_preprocess = use_arabert_preprocess
        self.lemmetize_data = lemmetize_data
        # in case no tokenizer is provided, we will use arabert latest tokenizer
        if tokenizer is None:
            # we will use the built model by aubmindlab (https://huggingface.co/aubmindlab/bert-base-arabertv2)
            model_name = 'aubmindlab/bert-base-arabertv2'
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = tokenizer

    def fit(self):
        """
        fitting a pre-process algorithm over the data. Fitting in many cases is not required and only a transform
        function is to be used. In cases some parameters should be fitted (e.g., learning stop-words) - such fit
        function is usable

        Parameters
        ----------

        :return:
        """
        pass

    def transform(self, sentences_list, keep_emojis=True, remove_RT_prefix=True, remove_punctuations=True):
        """
        transforming data and applying all pre-processing steps over it. In case 'fit' is required, it will yiled an
        error in case data is not fitted yet

        Parameters
        ----------
        :param sentences_list: list (of arabic sentences)
            list of sentences to apply the function on. Each sentence is treated independently
        :param keep_emojis: boolean. Default: False
            whether to keep the emojies in the text.
        :param remove_RT_prefix: boolean. Default: True
            whether to remove RT prefix in cases of retweets
        :param remove_punctuations: boolean. Default: True
            whether to remove all punctuations from the text (,?.$%# etc...)

        :return: list
            list of transformed arabic sentences. Same input list, but after the transform function has been applied
            over all of them

        """
        arabert_pre_process_obj = ArabertPreprocessor(model_name='bert-large-arabertv2', keep_emojis=keep_emojis,
                                                      remove_html_markup=True, replace_urls_emails_mentions=True)
        new_sentences_list = list()
        # looping over each sentence
        for cur_text in sentences_list:
            # in case we want to remove the RT in case of retweets
            if remove_RT_prefix and cur_text.startswith('"RT '):
                cur_text = cur_text[4:]
            if remove_RT_prefix and cur_text.startswith('RT '):
                cur_text = cur_text[3:]
            # in case we decided to use the arabert preprocess (it is a very good one)
            if self.use_arabert_preprocess:
                preprocessed_text = arabert_pre_process_obj.preprocess(cur_text)
                # in case we want to remove punctuations
                if remove_punctuations:
                    preprocessed_text_as_list = preprocessed_text.split(" ")
                    # removal of punctuation (e.g., '?', '!?!')
                    preprocessed_text_as_list = [cur_word for cur_word in preprocessed_text_as_list
                                                 if not all(j in string.punctuation for j in cur_word)]
                    new_sentences_list.append(' '.join(preprocessed_text_as_list))
                # in case we do not want to remove punctuations - we'll add the data as is
                else:
                    new_sentences_list.append(preprocessed_text)
            # currently not doing anything in such case, only supports the default case
            else:
                new_sentences_list.append(cur_text)
        return new_sentences_list

    @staticmethod
    def extract_emojis(text):
        """
        a simple function to handle emojies. It extracts emojies out of the text given as input.
        Note that both 're' and 'emoji' packages have to be installed

        function is taken from here - https://stackoverflow.com/questions/43146528/how-to-extract-all-the-emojis-from-text
        Parameters
        ----------
        :param text: str
            the text to handle
        :return: list
            list of emojies found. Empty list is returned in case no emojies in the text were found

        Examples
        --------
        >>> text_a ='🤔 🙈 me así, bla es, se 😌 ds 💕👭👙'
        >>> text_b = 'الفلسطيني لو حارب اسرئيل\U0001f974🤚هيمووت\nانما لما يحارب مصر😂🤚هياخد دعم وفلوس\nالقدس والله مش في دماغ الفلسطيني\nالقدس شماعه للمعونات والدعم والسلاح\nكلامي الاكثر عن قطاع غزه'
        >>>print(extract_emojis(text_a))
        ['🤔', '🙈', '😌', '💕', '👭', '👙']

        """
        emojis_list = map(lambda x: ''.join(x.split()), emoji.UNICODE_EMOJI.keys())
        r = re.compile('|'.join(re.escape(p) for p in emojis_list))
        aux = r.findall(text)
        return aux
