import pandas as pd
import os


class DataLoader(object):
    """ Class for loading data for any arabic NLP project

        Parameters
        ----------

        Attributes
        ----------
        files_path : str
            folder of the data location
            Currently the folder should include TSVs files to be loaded (later it can be other rather than tsvs)
        target_column_idx : int. Default: 0
            index of the target column (target=label)

        Examples
        --------
        >>> list_of_sentences = ['هذه جملة رائعة', 'ياله من يوم جميل 🤔 🙈']
        >>> preprocess_obj = PreProcess(use_default_farsa_preprocess=True)
        >>> preprocess_obj.transform(sentences_list=list_of_sentences, extract_and_paste_emojies=True)
        """
    def __init__(self, files_path, target_column_idx=0):
        self.files_path = files_path
        self.target_column_idx = target_column_idx

    def load_tsv_files(self, file_names, header=None, sample_rate=1.0, seed=1984, new_column_names=None):
        """
        a function to load tsv files into the project's environment.
        the function also allows sampling of the data and shuffle the data by default
        It returns a (pandas) dataframe for further modeling purposes

        Attributes
        ----------
        :param file_names: iterator
            iterator containing file names (including the .tsv suffix)
        :param header: int, list of int. Default: None
            row number(s) to use as the column names. It is the same parameter used in pandas read_table function.
            see - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_table.html
        :param sample_rate: float. Default: 1.0
            percentage of samples to be taken from the data. Such number should be in the [0,1] range.
            In case it is None - will be treated as 1.0
        :param seed: int
            a seed value to be used for sampling. Note that data is shuffled anyway
        :param new_column_names: list (of strings)
            the list of column names to initialize to the dataframe retunred by the function
        :return: dataframe
            a pandas dataframe with all data (and column names as given in new_column_names parameter)
        """
        all_datasets_loaded = list()
        for cur_file_name in file_names:
            cur_data_loaded = pd.read_table(os.path.join(self.files_path, cur_file_name), sep='\t', header=header)
            all_datasets_loaded.append(cur_data_loaded)
        # converting the list of DFs to a big DF
        full_data_df = pd.concat(all_datasets_loaded, ignore_index=True)
        # sampling the data, if not required to sample - we'll shuffle it
        if sample_rate is None:
            sample_rate = 1.0
        full_data_df = full_data_df.sample(frac=sample_rate, random_state=seed).reset_index(drop=True)
        if new_column_names is not None:
            full_data_df.columns = new_column_names
        return full_data_df

    def train_test_splitter(self):
        pass
