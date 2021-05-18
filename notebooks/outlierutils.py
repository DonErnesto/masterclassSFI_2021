import requests
import json
import functools

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score
cost_dict = {'kdd':
                {'FP':-25,
                'TP':500},
            'fraud':
                {'FP': -10,
                'TP':500},
            'pen':
                {'FP':-10,
                'TP':1000}
            }

def admin_only(func):
    """ Decorator for dangerous functions
    """
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        if not args[0].username == 'admin': #NB: first arg to an object-method is self
            print('Admin only!')
            return None
        if not kwargs.get('iknowwhatiamdoing', None):
            print('pass iknowwhatiamdoing=True to execute this')
            return None
        value = func(*args, **kwargs)
        return value
    return wrapper_decorator

class LabelSubmitter():
    def __init__(self, username, password, url='http://127.0.0.1:5000'):
        self.username = username
        self.password = password
        self.jwt_token = None
        self.base_url = url
        self.last_labels = None
        self.all_labels = None
        self._get_jwt_token()
        self._previous_score = None

    def _get_jwt_token(self):
        """ Posts to /auth
        """
        auth = requests.post(self.base_url + '/auth', json={"username": f"{self.username}",
                "password": f"{self.password}"})
        try:
            self.jwt_token = json.loads(auth.text)['access_token']
        except KeyError:
            print('Is the username and password correct?')
            return auth

    def post_predictions(self, idx, endpoint='pen'):
        """ Posts to /label
        sets self.last_labels
        """
        idx = [int(n) for n in idx] # replace numpy array and int64 by list with ints
        res = requests.post(url=self.base_url + '/label/{}'.format(endpoint),
                    json={'data': {'idx': idx}},
                   headers={'Authorization': 'JWT {}'.format(self.jwt_token)})
        self.res = res
        try:
            result = json.loads(res.text)['result']
            unzips = list(zip(*result))
            labels = pd.Series(index=unzips[0], data=unzips[1]).sort_index()
            self.last_labels = labels

            print(json.loads(res.text)['info'])
            N_tp = int(labels.sum())
            N_fp = int(len(labels) - N_tp)
            score = self._calculate_score(endpoint,
                                            N_tp=N_tp,
                                            N_fp=N_fp
                                            )
            precision = labels.mean()
            print('number of positives in submission: {:d}'.format(N_tp))
            print('precision of submission: {:.2%}'.format(precision))
            #print('current score: {}'.format(score))
            #print('previous score: {}'.format(self._previous_score))
            #self._previous_score = score
        except Exception as e:
            print(e)
            print(json.loads(res.text))

    def get_labels(self, endpoint='pen'):
        """ 'Gets' to /label
        sets self.all_labels
        """
        try:
            res = requests.get(url=self.base_url + '/label/{}'.format(endpoint),
                       headers={'Authorization': 'JWT {}'.format(self.jwt_token)})
            result = json.loads(res.text)['result']
            unzips = list(zip(*result))
            labels = pd.Series(index=unzips[0], data=unzips[1]).sort_index()
            self.all_labels = labels
            N_tp = int(labels.sum())
            N_fp = int(len(labels) - N_tp)
            score = self._calculate_score(endpoint,
                                            N_tp=N_tp,
                                            N_fp=N_fp
                                            )
            print('number of predictions made: {:d}'.format(int(len(labels))))
            print('total number of positives found: {:d}'.format(int(labels.sum())))
            print('total precision: {:.2%}'.format(labels.mean()))
            print('score: {}'.format(score))
            return labels
        except KeyError:
            print(json.loads(res.text))

    def get_statistics(self, endpoint='pen', plot=True):
        try:
            res = requests.get(url=self.base_url + '/labelstats/{}'.format(endpoint),
               headers={'Authorization': 'JWT {}'.format(self.jwt_token)})
            stats = json.loads(res.text)['result']
            stats_df = pd.DataFrame.from_dict(stats).T
            stats_df['precision'] = 100 * stats_df['N_positives_found'] / stats_df['N_submitted']
            if plot:
                fig, axs = plt.subplots(2, 1, figsize=(12,6))
                stats_df['N_submitted'].plot(kind='bar', ax=axs[0])
                stats_df['precision'].plot(kind='bar', ax=axs[1])
                axs[0].set_title('Number of submitted points')
                axs[1].set_title('Precision [%]')
                plt.tight_layout()
            return stats_df
        except KeyError:
            print(json.loads(res.text))

    def get_scores(self, endpoint='pen', plot=True, plot_only_active=True):
        try:
            res = requests.get(url=self.base_url + '/labelstats/{}'.format(endpoint),
               headers={'Authorization': 'JWT {}'.format(self.jwt_token)})
            stats = json.loads(res.text)['result']
            stats_df = pd.DataFrame.from_dict(stats).T
            stats_df = stats_df.rename(columns={'N_positives_found':
                        'N_true_positives'})

            stats_df['N_false_positives'] = stats_df['N_submitted'] - stats_df['N_true_positives']

            stats_df['score'] = self._calculate_score(
                            endpoint=endpoint,
                            N_tp=stats_df['N_true_positives'],
                            N_fp=stats_df['N_false_positives']
                                                    )
            if plot:
                fig, axs = plt.subplots(1, 1, figsize=(14,6))
                if plot_only_active:
                    stats_df = stats_df.loc[stats_df['N_submitted'] > 0, :]
                stats_df['score'].plot(kind='bar', ax=axs)
                axs.set_title('Score')
                plt.tight_layout()
            return stats_df
        except KeyError:
            return stats
            print(json.loads(res.text))


    def add_user(self, username, password):
        res = requests.post(url=self.base_url + '/newuser',
           headers={'Authorization': 'JWT {}'.format(self.jwt_token)},
            json={'username': username,
                          'password': password}
                           )
        print(json.loads(res.text))

    @staticmethod
    def _calculate_score(endpoint, N_tp, N_fp):
        cost_fp, cost_tp = cost_dict[endpoint]['FP'], cost_dict[endpoint]['TP']
        return (cost_fp * N_fp + cost_tp * N_tp)

    @admin_only
    def delete_user(self, username, iknowwhatiamdoing=False):
        res = requests.delete(url=self.base_url + '/removeuser/{}'.format(username),
           headers={'Authorization': 'JWT {}'.format(self.jwt_token)})
        try:
            print(json.loads(res.text))
        except:
            return res

    @admin_only
    def delete_labels(self, username, endpoint='pen', iknowwhatiamdoing=False):
        res = requests.delete(url=self.base_url + '/labeladmin/{}/{}'.format(
        endpoint, username),
           headers={'Authorization': 'JWT {}'.format(self.jwt_token)})
        try:
            print(json.loads(res.text))
        except:
            return res

def plot_outlier_scores(y_true, scores, title='', **kdeplot_options):
    """
    y_true (np-array): array with actual labels (0/1)
    scores (np-array): array with outlier scores
    title (str): title to be added to plot

    **kdeplot_options (such as bw for kde kernel width) are passed to sns.kdeplot()

    Returns: a pd.DataFrame with classification results
    """
    assert len(y_true) == len(scores), 'Error: '\
    'Expecting y_true and scores to be 1-D and of equal length'
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(scores, pd.Series):
        scores = scores.values
    assert len(y_true) == len(scores), 'y_true and scores should be of equal length'
    aucroc_score = roc_auc_score(y_true, scores)
    aucpr_score = average_precision_score(y_true, scores)
    classify_results = pd.DataFrame(data=pd.concat((pd.Series(y_true), pd.Series(scores)), axis=1))
    classify_results.rename(columns={0:'true', 1:'score'}, inplace=True)
    sns.kdeplot(classify_results.loc[classify_results.true==0, 'score'], label='negatives',
                shade=True, **kdeplot_options)
    sns.kdeplot(classify_results.loc[classify_results.true==1, 'score'], label='positives',
                shade=True, **kdeplot_options)
    plt.title('{} AUC-ROC: {:.3f}, AUC-PR: {:.3f}'.format(title, aucroc_score, aucpr_score))
    plt.xlabel('Predicted outlier score');
    return classify_results


def plot_top_N(y_true, scores, N=100):
    """
    y_true (np-array): array with actual labels (0/1)
    scores (np-array): array with outlier scores
    N (int): top-N size

    Returns: a pd.DataFrame with classification results

    """
    assert len(y_true) == len(scores), 'Error: '\
    'Expecting y_true and scores to be 1-D and of equal length'
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(scores, pd.Series):
        scores = scores.values
    N = min(N, len(scores))
    classify_results = pd.DataFrame(data=pd.concat((pd.Series(y_true), pd.Series(scores)), axis=1))
    classify_results.rename(columns={0:'true', 1:'score'}, inplace=True)
    classify_results = classify_results.sort_values(by='score', ascending=False)[:N]
    Npos_in_N = classify_results['true'].sum()

    fig, ax = plt.subplots(1, 1, figsize=(16, 2))
    ims = ax.imshow(np.reshape(classify_results.true.values, [1, -1]),
                extent=[-0.5, N, N/50, -0.5],
                vmin=0, vmax=1)
    ax.yaxis.set_visible(False)
    # ax.xaxis.set_ticklabels
    plt.colorbar(ims)
    plt.xlabel('Outlier rank [-]')
    plt.title(f'Yellow: positive, Purple:Negative. Number of positives found: {Npos_in_N} (P@Rank{N}: {Npos_in_N/N:.1%})')
    #plt.show()
    return classify_results

def median_imputation(df, median_impute_limit=0.95, impute_val=-999):
    """ inf/nan Values that occur more often than median_impute_limit are imputed with the median
    when less often, they are imputed by impute_val.
    Set median_impute_limit to 0 to always do median imputation
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.columns:
        if not df[col].dtype == 'object':
            mean_nan = df[col].isna().mean()
            if mean_nan > median_impute_limit: # then, impute by median
                df[col] = df[col].fillna(df[col].median())
            elif mean_nan > 0 and mean_nan <= median_impute_limit:
                df[col] = df[col].fillna(impute_val)

    return df


def reduce_mem_usage(df, verbose=True):
    """ function from Kaggle. Transforms the column data types to the smallest possible representation
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                #if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                #    df[col] = df[col].astype(np.int8)
                #elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                #    df[col] = df[col].astype(np.int16)
                if c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                #if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                #    df[col] = df[col].astype(np.float16)
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased from {:5.2f} to {:5.2f} Mb ({:.1f}% reduction)'.format(
        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
