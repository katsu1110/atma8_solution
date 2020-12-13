import numpy as np
import pandas as pd
import os
import sys
import gc
import pickle
import json
import tempfile
import pathlib
import itertools
from loguru import logger
from scipy import stats
from tqdm import tqdm
from sklearn import utils
from sklearn import preprocessing
from sklearn import decomposition
import texthero as hero
from nltk.util import ngrams

from fe_w2v import Category2VecWithW2V
from utils import reduce_mem_usage

# ------------------------------
# feature engineering functions
# ------------------------------
def user_score(df):
    df['is_tbd'] = 1 * (df['User_Score'] == 'tbd')
    df['User_Score'] = df['User_Score'].replace({'tbd': np.nan})
    df['User_Score'] = df['User_Score'].astype(float)
    for f in ['Critic_Score', 'Critic_Count', 'User_Score', 'User_Count']:
        df[f'isnan_{f}'] = 1 * df[f].isna()
        try:
            df[f] = df.groupby('Publisher')[f].transform(lambda x : x.fillna(x.mean()))
        except:
            try:
                df[f] = df.groupby('Platform')[f].transform(lambda x : x.fillna(x.mean()))
            except:
                df[f] = df[f].fillna(df[f].mean())
    return df

def rating(df):
    df['Rating'] = df['Rating'].map({'E': 4, 'T': 3, 'M': 2, 'E10+': 1})
    df['Rating'] = df['Rating'].fillna(0).astype(int)
    return df

def fill_genre(df):
    df['Genre'] = df['Genre'].fillna('Sports')
    return df

def fill_year(df):
    df['isnan_year'] = 1 * df['Year_of_Release'].isna()
    df['Year_of_Release'] = df.groupby('Platform')['Year_of_Release'].transform(lambda x : x.fillna(x.mean()))
    df['Year_of_Release'] = df['Year_of_Release'].astype(int)
    return df

def fill_developer(df):
    df['Developer'] = df['Developer'].fillna('Unknown')
    return df

def fill_publisher(df):
    df['Publisher'] = df['Publisher'].fillna("Unknown").astype(str)
    return df

def get_num(words):
    l = [w for w in str(words) if w.isdigit()]
    if len(l) == 0:
        return 0
    else:
        return float(l[-1])

def line_ngram(line, n=2):
    words = [w for w in line.split(' ') if len(w) != 0]
    return list(ngrams(words, n))

def easy_name(df):
    df['Name'] = df['Name'].apply(lambda x : str(x).replace(' I', ' 1').replace(' II', ' 2').replace(' III', ' 3').replace(' IV', ' 4'))
    df['Name_len'] = df['Name'].apply(lambda x : len(str(x)))
    df['Name_space_len'] = df['Name'].apply(lambda x : len(str(x).split(' '))-1)
    df['Name_stdlen'] = df['Name'].apply(lambda x : np.std([len(i) for i in str(x).split()]))
    df['Name_meanlen'] = df['Name'].apply(lambda x : np.mean([len(i) for i in str(x).split()]))
    df['Name_num'] = df['Name'].apply(lambda words: get_num(words))
    df['Name_is_num'] = 1 * (df['Name_num'] > 0)
    return df

def name2feats(df, column='Name', methods=['pca', 'kmeans', 'ngram']):
    # preprocessing
    custom_pipeline = [
                   hero.preprocessing.fillna
                   , hero.preprocessing.lowercase
                   , hero.preprocessing.remove_digits
                   , hero.preprocessing.remove_punctuation
                   , hero.preprocessing.remove_diacritics
                   , hero.preprocessing.remove_whitespace
                  ]
    df['clean_name'] = hero.clean(df[column], pipeline=custom_pipeline)

    # tfidf -> pca
    if 'pca' in methods:
        df['pca_name'] = hero.tfidf(df['clean_name'], max_features=200)
        df['pca_name'] = hero.pca(df['pca_name'], n_components=10)
        for i in np.arange(len(df['pca_name'].values[0])):
            df[f'tfidf_pca_{column}{i}'] = df['pca_name'].apply(lambda x : x[i])
        df.drop(columns=['pca_name'], inplace=True)

    # tfidf -> kmeans
    if 'kmeans' in methods:
        df[f'tfidf_kmeans_{column}'] = hero.tfidf(df['clean_name'], max_features=200)
        df[f'tfidf_kmeans_{column}'] = hero.kmeans(df[f'tfidf_kmeans_{column}'], n_clusters=10)

    # n-gram
    if 'ngram' in methods:
        for n in [2, 3]:
            name_grams = df[column].apply(lambda x : line_ngram(x, n))
            grams = [x for row in name_grams for x in row if len(x) > 0]
            top_grams = pd.Series(grams).value_counts().head(20).index
            df[f'{column}_in_top_{n}gram'] = name_grams.map(lambda x : any([i for i in x if i in top_grams]))

    df.drop(columns=['clean_name'], inplace=True)
    return df

def count_encoding(df, cat_feats):
    for c in cat_feats:
        mapper = df[c].value_counts()
        df[c + '_ce'] = df[c].map(mapper) / df.shape[0]
    return df

def quantile25(series):
   return series.quantile(0.25)

def quantile75(series):
   return series.quantile(0.75)

def flatten_column(df):
    df.columns = ["_".join(col).strip() for col in df.columns.values]
    renamer = {}
    for f in df.columns.values.tolist():
        if f.endswith('_'):
            renamer[f] = f[:-1]
    df.rename(columns=renamer, inplace=True)
    return df

def row_feats(df):
    df['Critic_score_mean'] = df['Critic_Score'] / (df['Critic_Count'] + 1)
    df['User_score_mean'] = df['User_Score'] / (df['User_Count'] + 1)
    df['Critic_per_user'] = df['Critic_Count'] / (df['User_Count'] + 1)
    df['Critic_score_per_user'] = df['Critic_Score'] / (df['User_Count'] + 1)
    df['User_score_per_critic'] = df['User_Score'] / (df['Critic_Count'] + 1)
    df['Critic_User_score_ratio'] = df['Critic_Score'] / (df['User_Score'] + 1)
    df['Critic_score_x_count'] = df['Critic_Score'] * df['Critic_Count']
    df['User_score_x_count'] = df['User_Score'] / df['User_Count']
    df['Critic_x_user_count'] = df['Critic_Count'] * df['User_Count']
    df['Critic_x_user_score_count'] = df['Critic_Score'] * df['User_Count']
    df['User_x_critic_score_count'] = df['User_Score'] * df['Critic_Count']
    df['Critic_User_x_score'] = df['Critic_Score'] * df['User_Score']
    return df

def cat_by_cat(df, cat=['Name', 'Platform', 'Genre', 'Year_of_Release', 'Developer', 'Rating', 'Publisher', ]):
    for c in cat:
        feats = [f for f in cat if f not in [c, ]]
        if c in ['Name',]:
            feats = ['Platform', 'Year_of_Release']
        elif c in ['Developer', 'Publisher']:
            feats = ['Platform', 'Year_of_Release', 'Genre', 'Rating']
        elif c in ['Platform_Genre']:
            feats = [f for f in feats if f not in ['Platform', 'Genre']]
        elif c in ['Platform', 'Genre']:
            feats = [f for f in feats if f not in ['Platform_Genre']]
        agg_df = df.groupby(c)[feats].nunique().reset_index()
        df = df.merge(agg_df, how='left', on=c, suffixes=('', f'_nunique_by_{c}'))
    return df

def add_transformed(agg_df, trans, v, method='lda'):
    d = trans.fit_transform(agg_df.values)
    pca_df = pd.DataFrame()
    pca_df[f'{v[0]}'] = agg_df.index.values
    for i in range(d.shape[1]):
        f = f'{method}{i+1}_{v[0]}_{v[1]}'
        pca_df[f] = d[:, i]
    return pca_df

def lda(df, cat=['Name', 'Platform', 'Genre', 'Developer', 'Year_of_Release', 'Rating', ]):
    for v in tqdm(itertools.permutations(cat, 2)):
        # co-occurence matrix
        n_comp = 3
        if (v[0] in ['Publisher', 'Name', 'Developer']) & (v[1] in ['Publisher', 'Name', 'Developer', 'Rating']):
            continue
        if (v[0] == 'Platform_Genre') & (v[1] in ['Platform', 'Genre']):
            continue
        if (v[1] == 'Platform_Genre') & (v[0] in ['Platform', 'Genre']):
            continue
        if f'lda1_{v[0]}_{v[1]}' not in df.columns.values.tolist():
            print(f'{v[0]} vs {v[1]}')
            agg_df = pd.crosstab(df[v[0]], df[v[1]])

            # lda
            trans = decomposition.LatentDirichletAllocation(n_components=n_comp, random_state=42)
            trans2 = decomposition.NMF(n_components=n_comp, max_iter=8000, random_state=42)
            trans3 = decomposition.PCA(n_components=n_comp, random_state=42)

            lda_df = add_transformed(agg_df, trans, v, method='lda')
            nmf_df = add_transformed(agg_df, trans2, v, method='nmf')
            pca_df = add_transformed(agg_df, trans3, v, method='pca')

            # merge
            df = df.merge(lda_df, how='left', on=v[0])
            df = df.merge(nmf_df, how='left', on=v[0])
            df = df.merge(pca_df, how='left', on=v[0])
    return df

def w2v(df, cats=['Platform'], n_components=5):
    c = Category2VecWithW2V(cats, 
                 n_components=n_components, min_count=1, workers=4, seed=777, 
                 save_model_path=None, name='category2vec')
    df = c.transform(df)
    return df

def agg_feats(df, keys=['Platform', 'Genre', 'Rating', ], nums=['User_Score']):
    agg_methods = ['mean', 'std', 'max', 'min', pd.DataFrame.mad, quantile25, quantile75]
    for k in tqdm(keys):
        if k == 'Year_of_Release':
            nums = [n for n in nums if n != 'Year_of_Release']
        agg_df = df.groupby(k)[nums].agg(agg_methods).reset_index()
        agg_df = flatten_column(agg_df)

        # more
        if k != 'Publisher':
            for n in nums:
                agg_df[n + '_ff'] = agg_df[n + '_std'] / agg_df[n + '_mean']
                agg_df[n + '_maxmin_ratio'] = agg_df[n + '_min'] / agg_df[n + '_max']
                agg_df[n + '_q25_75_ratio'] = agg_df[n + '_quantile25'] / agg_df[n + '_quantile75']
                agg_df[n + '_maxmin_diff'] = agg_df[n + '_max'] - agg_df[n + '_min']
                agg_df[n + '_q25_75_diff'] = agg_df[n + '_quantile75'] - agg_df[n + '_quantile25']

        # merge
        agg_df.columns = [k] + [f + f'_{k}' for f in agg_df.columns.values[1:].tolist()]
        df = df.merge(agg_df, how='left', on=k)

        # further more
        if k != 'Publisher':
            for n in nums:
                df[n + f'_diff2mean_{k}'] = df[n] - df[n + f'_mean_{k}']
                df[n + f'_ratio2mean_{k}'] = df[n] / df[n + f'_mean_{k}']
                df[n + f'_diff2max_{k}'] = df[n] - df[n + f'_max_{k}']
                df[n + f'_ratio2max_{k}'] = df[n] / df[n + f'_max_{k}']
                df[n + f'_diff2min_{k}'] = df[n] - df[n + f'_min_{k}']
                df[n + f'_ratio2min_{k}'] = df[n] / df[n + f'_min_{k}']
                df[n + f'_z_{k}'] = df[n + f'_diff2mean_{k}'] / df[n + f'_std_{k}']
    return df

def non_overlap_category(train, test, cat_feats):
    for c in cat_feats:
        intersect = list(set(train[c].values.tolist()) & set(test[c].values.tolist()))
        renamer = {}
        for v, i in enumerate(intersect):
            renamer[i] = int(v)
        train[c] = train[c].map(renamer).fillna(999).astype(int)
        test[c] = test[c].map(renamer).fillna(999).astype(int)
    return train, test

def preprocess(train, test):
    # -------------------------
    # config
    # -------------------------
    target = 'Global_Sales'
    non_targets = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
    group = 'Publisher'
    outs = ['Developer', 'Name']
    cat_feats = ['Platform', 'Genre', 'Rating', 'Platform_Genre']
    
    # -------------------------
    # add id
    # -------------------------
    train['data_id'] = np.arange(len(train))
    test['data_id'] = np.arange(len(test)) + len(train)
    assert train['data_id'].values[-1] + 1 == test['data_id'].values[0]
    train['data_id'] = train['data_id'].astype(int)
    test['data_id'] = test['data_id'].astype(int)

    # -------------------------
    # preprocess
    # -------------------------
    logger.debug('preprocessing...')

    # nan counts
    train['num_nans'] = train.isnull().sum(axis=1)
    test['num_nans'] = test.isnull().sum(axis=1)

    # remove non-overlapped platforms
    train.loc[train['Platform'].isin(['GG', 'SCD', 'PCFX']), 'Platform'] = 'TG16'
    assert train['Platform'].nunique() == test['Platform'].nunique()

    # concat and preprocess for the whole df
    df = pd.concat([train, test], ignore_index=True)
    df = fill_year(df)
    df = fill_developer(df)
    df = fill_genre(df)
    df['Platform_Genre'] = df['Platform'].astype(str) + df['Genre'].astype(str)
    df = user_score(df)
    df = fill_publisher(df)
    df = easy_name(df)

    # ----------------------------
    # feature engineering
    # ----------------------------
    logger.debug('count encoding...')
    df = count_encoding(df, cat_feats)
    
    logger.debug('cats nunique...')
    df = cat_by_cat(df, ['Platform', 'Genre', 'Rating', 'Platform_Genre', 'Year_of_Release', group] + outs)

    logger.debug('tfidf etc...')
    df = name2feats(df, column='Name', methods=['pca', 'kmeans', 'ngram', ])

    logger.debug('lda...')
    df = lda(df, ['Platform', 'Genre', 'Rating', 'Year_of_Release', 'Platform_Genre', group] + outs)
    
    logger.debug('agg...')
    df = row_feats(df)
    df = rating(df)
    nums = ['Critic_Score', 'Critic_Count', 'User_Count', 'User_Score', 'Critic_score_mean', 'User_score_mean', 
        'Critic_User_score_ratio', 'Critic_per_user','Year_of_Release']
    df = agg_feats(df, ['Platform', 'Genre', 'Rating', 'Year_of_Release', 'Platform_Genre', group], nums)
    
    logger.debug('w2v...')
    df = w2v(df, ['Platform', 'Genre', 'Rating', 'Year_of_Release', group] + outs, n_components=2)
    df = reduce_mem_usage(df)

    # group to int
    df[group] = df[group].fillna('NaN')
    df[group] = preprocessing.LabelEncoder().fit_transform(df[group])
    df[group] = df[group].astype(int)

    # split
    train = df.loc[df['data_id'].isin(train['data_id']), :].sort_values(by='data_id').reset_index(drop=True)
    test = df.loc[df['data_id'].isin(test['data_id']), :].sort_values(by='data_id').reset_index(drop=True)
    drops = [group, target, ] + outs
    features = [f for f in test.columns.values.tolist() if f not in drops]

    # remove non-overlap of cats
    train, test = non_overlap_category(train, test, cat_feats)
    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)

    # remove too many nan feats
    new_feats = []
    for f in features:
        try:
            variance = test[f].var()
            nnan = test[f].isna().sum()
        except:
            logger.debug(f'ERR: {f}')
            continue
        if (nnan < (test.shape[0] // 1.5)) & (variance > 0.001):
            new_feats.append(f)
        else:
            logger.debug('{} dropped...nnan={:,}, variance={:.3f}'.format(f, nnan, variance))
    train = train[new_feats + [target, group] + non_targets]
    for t in [target] + non_targets:
        train[t] = train[t].astype(float)
    test = test[new_feats + [group]]
    cat_feats = [f for f in cat_feats if f in new_feats]

    # final confirmation 
    train = train.sort_values(by='data_id').reset_index(drop=True)
    test = test.sort_values(by='data_id').reset_index(drop=True)

    return train, test, new_feats, cat_feats, target, group