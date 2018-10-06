import os
import tarfile
import pandas as pd
import numpy as np
from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit
import hashlib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    # this returns a Pandas.Dataframe object which is a basically a 2d array
    return pd.read_csv(csv_path)


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# this is not an ideal function as it will return different training data each time which will make the function too
# general
def split_train_test(data, test_ratio):
    # creates a list of numbers ranging from 1 - the length of the data object, in random order
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    # iloc selects rows based on a list of numbers given, representing which rows to select from the dataframe object
    # test
    return data.iloc[train_indices], data.iloc[test_indices]


def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]


'''we may want to evenly distribute the data around some important attribute
therefore we can create discrete categories around one attribute eg medium income and make sure our test and train data
has an equal distribution from all the defined categories'''


def create_median_income_categories(housing):
    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
    return housing


''' Here we use the scikit learn's StratifiedShuffleSplit to make sure we have an even proportion of variables of some
category, in this case the income category which we compute in create_median_income_categories, in the train and test
sets as is in the overall data set. '''


def create_stratified_shuffle(housing):
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]




    



