import datetime

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pathlib
from nltk.stem import PorterStemmer
from sklearn import model_selection, naive_bayes, svm
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, \
    adjusted_mutual_info_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


def get_data_bbc_articles() -> pd.DataFrame:
    """
    data is taken from: http://mlg.ucd.ie/datasets/bbc.html
    :return: the dataset containing labeled BBC articles
    """
    # local path where text files were placed
    path = pathlib.Path('/home/andrei/temp/bbcsport')
    data = load_files(path, encoding="utf-8", decode_error="replace", random_state=500)

    # remove newlines
    data['data'] = [it.lower().replace('\n\n', ' ') for it in data['data']]

    df = pd.DataFrame(list(zip(data['data'], data['target'])), columns=['text', 'label'])
    return df


def get_data_sms_spam():
    # data is taken from: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
    file_path = '/home/andrei/temp/SMSSpamCollection'

    # read table from text file
    data = pd.read_table(file_path,
                         sep='\t',
                         header=None,
                         names=['label', 'text'])
    return data


def get_data_fake_news():
    # data is taken from: https://zenodo.org/record/2607278
    file_path = '../../playground/FA-KES-Dataset2.csv'

    # read table from text file
    data = pd.read_csv(file_path, delimiter=',')

    # extract only text and labels
    res = pd.DataFrame()
    res['text'] = data['article_title'] + ' ' + data['article_content']
    res['label'] = data['labels']

    return res


def get_toxic_comments():
    # data is taken from: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
    file_path = '/home/andrei/temp/toxic_text/train.csv'

    # read table from text file
    data = pd.read_csv(file_path, delimiter=',')

    # extract only text and labels
    res = pd.DataFrame()
    res['text'] = data['comment_text']
    res['label'] = data['toxic'] + data['severe_toxic'] + data['obscene'] + data['threat'] + \
                   data['insult'] + data['identity_hate']

    return res


def get_huffpost_news():
    # data is taken from: https://www.kaggle.com/rmisra/news-category-dataset/download
    file_path = '/home/andrei/temp/huffpost_dataset/News_Category_Dataset_v2.json'

    columns = ['category', 'headline', 'authors', 'link', 'short_description', 'date']
    # data = pd.DataFrame(columns=columns)
    #
    # # read data from json file
    # with open(file_path) as f:
    #     lines = f.readlines()
    #     # select only the first 10k news items
    #     for idx, line in enumerate(lines):
    #         if (idx % 1000) == 0:
    #             print('Read {}/{} lines.'.format(idx, len(lines)))
    #         line_data = json.loads(line)
    #         line_data_df = pd.DataFrame.from_dict(line_data, orient='index').T
    #         data = data.append(line_data_df, ignore_index=True)
    #
    # data.to_csv('./huff_news.csv')

    data = pd.read_csv('../../playground/huff_news.csv')

    # create response dataframe
    res = pd.DataFrame(columns=['text', 'label'])

    categories = sorted(set(data.category))

    # reduce number of labels to 5 and select 100 from each
    cnt = 0
    for category in categories[0:10]:
        sel_data = data.loc[data['category'].isin([category])]
        if len(sel_data) > 2000:
            data_for_label = data.loc[data['category'].isin([category])].head(2000)
            data_for_label = data_for_label[~data_for_label.short_description.isna()][0:1000]
            tmp = pd.DataFrame(columns=['text', 'label'])
            tmp['text'] = data_for_label.headline + ' ' + data_for_label.short_description
            tmp['label'] = data_for_label.category
            res = res.append(tmp, ignore_index=True)
            cnt += 1

        if cnt == 5:
            break

    return res


def compute_other_scores(_x_test: pd.Series, _y_test: np.ndarray, y_pred: np.ndarray):
    """
    Compute various precision scores.
    :param _x_test: testset X
    :param _y_test: actual testset Y
    :param y_pred: predicted testset Y
    :return:
    """

    # A clustering result satisfies homogeneity if all of its clusters contain only
    # data points which are members of a single class.
    _homogeneity_score = homogeneity_score(_y_test, y_pred)

    # A clustering result satisfies completeness if all the data points that are members
    # of a given class are elements of the same cluster.
    _completeness_score = completeness_score(_y_test, y_pred)

    # The V-measure is the harmonic mean between homogeneity and completeness
    _v_measure_score = v_measure_score(_y_test, y_pred)

    # The Rand Index computes a similarity measure between two clusterings
    #     by considering all pairs of samples and counting pairs that are
    #     assigned in the same or different clusters in the predicted and
    #     true clusterings.
    _adjusted_rand_score = adjusted_rand_score(_y_test, y_pred)

    # Adjusted Mutual Information (AMI) is an adjustment of the Mutual
    #     Information (MI) score to account for chance.
    _adjusted_mutual_info_score = adjusted_mutual_info_score(_y_test, y_pred)

    # The silhouette value is a measure of how similar an object is to its own cluster (cohesion) compared
    # to other clusters (separation). It is only applicable for geometric classification algorithms
    _silhouette_score = 0.0  # silhouette_score(_x_test, y_pred, metric='euclidean')
    print('homogeneity_score: {},\n completeness_score: {},\n '
          'v_measure_score: {},\n adjusted_rand_score: {},\n '
          'adjusted_mutual_info_score: {},\n silhouette_score: {}.\n'.format(_homogeneity_score,
                                                                             _completeness_score,
                                                                             _v_measure_score,
                                                                             _adjusted_rand_score,
                                                                             _adjusted_mutual_info_score,
                                                                             _silhouette_score))


def read_dataset(_dataset_type: str = 'custom') -> pd.DataFrame:
    """
    Read dataset based on its type.
    :param _dataset_type:
    :return:
    """
    # get data based on type
    if _dataset_type == 'custom':
        _dataset = pd.read_csv(r"../../playground/corpus.csv", encoding='latin-1')
    elif _dataset_type == 'sms':
        _dataset = get_data_sms_spam()
    elif _dataset_type == 'bbc':
        _dataset = get_data_bbc_articles()
    elif _dataset_type == 'fakes':
        _dataset = get_data_fake_news()
    elif _dataset_type == 'toxic':
        _dataset = get_toxic_comments()
    elif _dataset_type == 'huff':
        _dataset = get_huffpost_news()
    else:
        raise RuntimeError('Unknown dataset type! try again..')

    print('Started loading dataset "{}"'.format(_dataset_type))

    # convert the labels from strings to binary values for our classifier
    _dataset['label'] = _dataset.label.map({l: idx for idx, l in enumerate(set(_dataset.label))})

    # convert all characters to lower-case
    _dataset['text'] = _dataset.text.map(lambda x: x.lower())

    # remove any punctuation
    _dataset['text'] = _dataset.text.str.replace('[^\w\s]', '')

    # apply tokenization: divide a string into substrings by splitting on the specified string (defined in subclasses).
    _dataset['text'] = _dataset['text'].apply(nltk.word_tokenize)

    # select only words with a length smaller than 50
    _dataset['text'] = _dataset['text'].apply(lambda l: [it for it in l if len(it) <= 50])

    # perform word stemming. The idea of stemming is to normalize our text to remove all variations of words carrying
    # the same meaning, regardless of the tense. Porter Stemmer stemming algorithm is the most popular:
    stemmer = PorterStemmer()
    _dataset['text'] = _dataset['text'].apply(lambda x: [stemmer.stem(y) for y in x])

    print('Done loading dataset "{}"'.format(_dataset_type))
    return _dataset


def truncate_dataset(dataset: pd.DataFrame, _max_words_per_document: int = 100,
                     max_texts: int = 1000) -> pd.DataFrame:
    """
    This method does 2 things: 1. select a random sample from the dataset
                               2. reduce the amount of words in a document
    :param dataset: input dataset
    :param _max_words_per_document: the maximum allowed number of words per document
    :param max_texts: maximum allowed number of text from the dataset
    :return: the truncated version of the dataset
    """
    if _max_words_per_document:
        dataset['text'] = dataset['text'].apply(lambda x: ' '.join(x[0:_max_words_per_document]))
    else:
        dataset['text'] = dataset['text'].apply(lambda x: ' '.join(x))

    # select a random subset of the texts
    dataset = dataset.sample(max_texts)

    return dataset


def run(_dataset: pd.DataFrame, _test_size: float = 0.3, _max_features: int = None, _count_based=False):
    """
    Do the actual training, fitting and testing of the models.
    :param _dataset: input dataset
    :param _test_size: how large the test is
    :param _max_features: maximum number of features
    :param _count_based: weather it's based on TF-IDF or on counts
    :return:
    """

    x_test, x_train, y_test, y_train = split_and_convert(_dataset, _test_size, _max_features, _count_based)

    # 1. Bayes
    _bayes_accuracy_score, y_pred_bayes = classify_bayes(x_test, x_train, y_test, y_train)

    # 2. SVM
    _svm_accuracy_score, y_pred_svm = classify_svc(x_test, x_train, y_test, y_train)

    # 3. Decision Tree
    _dtree_accuracy_score, y_pred_dtree = classify_dtree(x_test, x_train, y_test, y_train)

    compute_other_scores(x_test, y_test, y_pred_bayes)
    compute_other_scores(x_test, y_test, y_pred_svm)
    compute_other_scores(x_test, y_test, y_pred_dtree)

    print("Naive Bayes Accuracy Score -> ", _bayes_accuracy_score)
    print("SVM Accuracy Score -> ", _svm_accuracy_score)
    print("Decision Tree Accuracy Score -> ", _dtree_accuracy_score)

    return _bayes_accuracy_score, _svm_accuracy_score, _dtree_accuracy_score


def split_and_convert(_dataset, _test_size=0.3, _max_features=None, _count_based=False):
    if _count_based:
        # print('Running based on counts')
        from sklearn.feature_extraction.text import CountVectorizer
        count_vect = CountVectorizer()
        counts = count_vect.fit_transform(_dataset['text'])
        # We could leave it as the simple word - count per message, but it is better to use
        # Term Frequency Inverse Document Frequency, more known as tf - idf:
        from sklearn.feature_extraction.text import TfidfTransformer
        transformer = TfidfTransformer().fit(counts)
        text = transformer.transform(counts)
    else:
        # print('Running based on TF-IDF')
        text = _dataset['text']

    # split the model into train and test datasets
    x_train, x_test, y_train, y_test = model_selection.train_test_split(text, _dataset['label'], test_size=_test_size)
    # encode (replace) text labels with numerical values between 0 and n_classes-1
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.fit_transform(y_test)
    if not _count_based:
        # vectorize words using TF-IDF in order to find word importance within the dataset
        tfidf_vect = TfidfVectorizer(max_features=_max_features)
        tfidf_vect.fit(_dataset['text'])

        # transform training and test datasets
        x_train = tfidf_vect.transform(x_train).toarray()
        x_test = tfidf_vect.transform(x_test).toarray()
    return x_test, x_train, y_test, y_train


def classify_dtree(x_test, x_train, y_test, y_train):
    # create Decision tree classifier
    model_dtree = DecisionTreeClassifier(random_state=0)
    # fit Decision tree classifier on training set
    model_dtree.fit(x_train, y_train)
    # predict on test dataset
    y_pred_dtree = model_dtree.predict(x_test)
    # Use 'accuracy_score()' function to compute the prediction accuracy
    _dtree_accuracy_score = accuracy_score(y_pred_dtree, y_test) * 100
    return _dtree_accuracy_score, y_pred_dtree


def classify_svc(x_test, x_train, y_test, y_train):
    # crate SVM classifier - 'linear' model performs better than 'rbf' model
    model_svm = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    # fit SVM classifier on training set
    model_svm.fit(x_train, y_train)
    # predict on test dataset
    y_pred_svm = model_svm.predict(x_test)
    # Use 'accuracy_score()' function to compute the prediction accuracy
    _svm_accuracy_score = accuracy_score(y_pred_svm, y_test) * 100
    return _svm_accuracy_score, y_pred_svm


def classify_bayes(x_test, x_train, y_test, y_train):
    # create Naive Bayes classifier
    # model_bayes = naive_bayes.MultinomialNB()  # the algorithm assumes that the features are drawn from a multinomial distribution
    model_bayes = naive_bayes.GaussianNB()  # continuous values associated with each class are distributed according to a normal (or Gaussian) distribution
    # model_bayes = naive_bayes.BernoulliNB()  # designed for binary/boolean features.
    # model_bayes = naive_bayes.ComplementNB()  # designed for imbalanced datasets
    # fit Bayesian classifier on training set
    model_bayes.fit(x_train, y_train)
    # predict on test dataset
    y_pred_bayes = model_bayes.predict(x_test)
    # compute accuracy score by using accuracy_score() function
    _bayes_accuracy_score = accuracy_score(y_pred_bayes, y_test) * 100
    return _bayes_accuracy_score, y_pred_bayes


def plot_results(_results):
    plt.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.rc('legend', fontsize=23, loc='lower right')

    # unstack the dataframe
    _results.groupby(['max_no_texts', 'classifier_type']).accuracy.agg(['median']).unstack().plot(ax=ax, grid=True,
                                                                                                  kind='bar')

    ax.legend(['DTree', 'SVC', 'Bayes'])
    ax.set_xlabel('Dataset length', size=23)
    ax.set_ylabel('Accuracy [%]', size=23)


if __name__ == "__main__":
    # set seed to a constant value to avoid inconsistencies between various runs
    np.random.seed(500)

    # set max features
    max_features = [None]

    # max number of words / line
    max_words_per_document = [None]

    # maximum number of texts to be selected from the dataset
    max_no_texts = range(25, 525, 25)  # [50, 100, 150, 200, 250]

    # size of the test [%]
    test_size = 0.3

    # number of experiments for each parameter combination
    exp_count = 100

    # set if it should be based on word counting or TF-IDF
    count_based = False

    # define dataset type: custom/sms/bbc/fakes
    # dataset_name = 'huff'  # ComplementNB()
    dataset_name = 'bbc'  # ComplementNB()

    dataset = read_dataset(_dataset_type=dataset_name)

    results = pd.DataFrame(columns=['dataset_type', 'max_features', 'max_words_per_line',
                                    'max_no_texts', 'classifier_type', 'exp_no', 'accuracy'])

    # do parametric sweeping
    for mf in max_features:
        for mwl in max_words_per_document:
            for mnt in max_no_texts:
                for exp_no in range(exp_count):
                    print('Running experiment for {} max_words_per_line, '
                          '{} max_no_texts, {} max_features, exp_no: {}'.format(mwl, mnt, mf, exp_no))
                    trauncated_dataset = truncate_dataset(dataset=dataset.copy(), _max_words_per_document=mwl,
                                                          max_texts=mnt)
                    bayes_accuracy_score, svc_accuracy_score, dtree_accuracy_score = run(_dataset=trauncated_dataset,
                                                                                         _test_size=test_size,
                                                                                         _max_features=mf,
                                                                                         _count_based=count_based)
                    results = results.append({'dataset_type': dataset_name, 'max_features': mf,
                                              'max_words_per_line': mwl,
                                              'max_no_texts': mnt, 'classifier_type': 'bayes',
                                              'exp_no': exp_no, 'accuracy': bayes_accuracy_score}, ignore_index=True)
                    results = results.append({'dataset_type': dataset_name, 'max_features': mf,
                                              'max_words_per_line': mwl,
                                              'max_no_texts': mnt, 'classifier_type': 'SVC',
                                              'exp_no': exp_no, 'accuracy': svc_accuracy_score}, ignore_index=True)
                    results = results.append({'dataset_type': dataset_name, 'max_features': mf,
                                              'max_words_per_line': mwl,
                                              'max_no_texts': mnt, 'classifier_type': 'DTree',
                                              'exp_no': exp_no, 'accuracy': dtree_accuracy_score}, ignore_index=True)

    results_file_path = './results_{}.csv'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
    print('Storing results to "{}"'.format(results_file_path))
    results.to_csv(results_file_path)
    print(results.head())
