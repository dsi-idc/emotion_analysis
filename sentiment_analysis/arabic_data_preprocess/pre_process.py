from transformers import AutoTokenizer
from arabert.preprocess_arabert import preprocess, never_split_tokens
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
        use_default_farsa_preprocess : boolean. Default: True
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
    def __init__(self, use_default_farsa_preprocess=True, tokenizer=None, lemmetize_data=False):
        self.use_default_farsa_preprocess = use_default_farsa_preprocess

        self.lemmetize_data = lemmetize_data
        # in case tokenizer, we will use arabert tokenizer
        if tokenizer is None:
            # we will use the built model by aubmindlab (https://huggingface.co/aubmindlab/bert-base-arabert)
            model_name = 'aubmindlab/bert-base-arabert'
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

    def transform(self, sentences_list, extract_and_paste_emojies=False):
        """
        transforming data and applying all pre-processing steps over it. In case 'fit' is required, it will yiled an
        error in case data is not fitted yet

        Parameters
        ----------
        :param sentences_list: list (of arabic sentences)
            list of sentences to apply the function on. Each sentence is treated independently
        :param extract_and_paste_emojies: boolean. Default: False
            whether to handle emojies is a special way. Currently we only know how to handle emojies in a very specific
            way (extract them and then paste them at the end of the sentence - not ideal)
            TODO: handle empjies in a better way (do not convert them to ?? and "leave" them as is in the sentence)

        :return: list
            list of transformed arabic sentences. Same input list, but after the transform function has been applied
            over all of them

        """
        farasa_segmenter = FarasaSegmenter(interactive=True)
        new_sentences_list = list()
        # looping over each sentence
        for cur_text in sentences_list:
            # in case we decided to use the farase preprocess
            if self.use_default_farsa_preprocess:
                preprocessed_text = preprocess(cur_text, do_farasa_tokenization=True,
                                               farasa=farasa_segmenter, use_farasapy=True)
                preprocessed_text_as_list = preprocessed_text.split(" ")
                # removal of punctuation (e.g., '?', '!?!')
                preprocessed_text_as_list = [cur_word for cur_word in preprocessed_text_as_list
                                             if not all(j in string.punctuation for j in cur_word)]
                if extract_and_paste_emojies:
                    emojies_found = self.extract_emojis(text=cur_text)
                    preprocessed_text_as_list.extend(emojies_found)
                new_sentences_list.append(' '.join(preprocessed_text_as_list))
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
