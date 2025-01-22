#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 08:13:23 2024

@author: cmheilig
"""

#%% Set up environment
import os
import json
import zipfile
from pandas.api.types import CategoricalDtype
import pandas as pd
from tqdm import tqdm
import pprint as pp

os.chdir('C:/DataPulse')

#%% Reconstruct metadata for mirror and for corpus

## Information from zip archives of HTML mirrors
# mirror_df = pd.read_csv('cdc_mirror_list.csv')
# Series,Length,Method,Size,Cmpr,Date    Time,CRC-32,Name
# ['Series', 'Length', 'Method', 'Size', 'Cmpr', 'Date    Time', 'CRC-32', 'Name']

# process the Altmeric top 10 MMWR journals by year
PROCESS_ALTMERIC_JOURNAL = True

# Year to process
YEAR = 2010

mirror_df = pd.read_csv(
    'C:/DataPulse/csv-output/cdc_mirror_list.csv',
    header=0, 
    names=['series', 'file_size', 'compress_type', 
           'compress_size', 'compress_pct', 'date_time', 'CRC', 
           'filename'],
    parse_dates=['date_time'], 
    dtype={'series': 'category', 'file_size': 'int', 'compress_type': 'category', 
           'compress_size': 'int', 'compress_pct': 'str', 'CRC': 'string', 
           'filename': 'string'})
mirror_df['compress_pct'] = pd.to_numeric(mirror_df.compress_pct.str[:-1])

## Information about processed corpus collections
series_cat = CategoricalDtype(
    ['mmwr', 'mmnd', 'mmrr', 'mmss', 'mmsu', 'eid', 'eid0', 'eid1', 'eid2',
     'pcd'], ordered=True)
level_cat = CategoricalDtype(
    ['home', 'series', 'volume', 'issue', 'article'], ordered=True)
stratum_cat = CategoricalDtype(
    ['mmwr_toc', 'mmwr_art', 'eid_toc', 'eid_art', 'pcd_toc', 'pcd_art'], 
    ordered=True)
collection_cat = CategoricalDtype(
    ['mmwr_toc_en', 'mmrr_toc_en', 'mmss_toc_en', 'mmsu_toc_en', 
     'mmwr_art_en', 'mmrr_art_en', 'mmss_art_en', 'mmsu_art_en', 
     'mmnd_art_en', 'mmwr_art_es', 
     'eid_toc_en', 'eid0_art_en', 'eid1_art_en', 'eid2_art_en', 
     'pcd_toc_en', 'pcd_toc_es', 'pcd_art_en', 'pcd_art_es', 
     'pcd_art_fr', 'pcd_art_zhs', 'pcd_art_zht'], ordered=True)

# pd.read_csv('cdc_corpus_df.csv').columns
# ['url', 'stratum', 'collection', 'series', 'level', 'lang', 'dl_year_mo', 'dl_vol_iss', 'dl_date', 'dl_page', 'dl_art_num', 'dateline', 'base', 'string', 'link_canon', 'mirror_path', 'md_citation_doi', 'title', 'md_citation_categories', 'dl_cat', 'md_kwds', 'md_desc', 'md_citation_author']

corpus_df = pd.read_csv(
    'C:/DataPulse/csv-output/cdc_corpus_df.csv',
    # parse_dates=['datetime'], 
    dtype={'url': 'string', 
           'stratum': stratum_cat, 'collection': collection_cat, 
           'series': series_cat, 'level': level_cat, 
           'lang': 'category', 'dl_year_mo': 'category', 
           'dl_vol_iss': 'category', 'dl_date': 'string', 'dl_page': 'string', 
           'dl_art_num': 'string', 'dateline': 'string', 'base': 'string', 
           'string': 'string', 'link_canon': 'string', 'mirror_path': 'string', 
           'md_citation_doi': 'string', 'title': 'string', 
           'md_citation_categories': 'string', 'dl_cat': 'string', 
           'md_kwds': 'string', 'md_desc': 'string', 
           'md_citation_author': 'string'})
# pd.to_datetime(corpus_df.dl_date, format='ISO8601'): YYYY-MM -> YYYY-MM-01


#%% Reconstruct contents extracted from corpus

# Location of zipped JSON collections

json_out_dir = 'C:/DataPulse/json-outputs'

# Sample code to interrogate zip archive
# with zipfile.ZipFile(json_out_dir + 'txt/mmsu_toc_en_txt_json.zip', 'r') as json_zip:
#     json_zip.printdir()
#     print(json_zip.namelist())
#     print(json_zip.infolist())
#     print(json_zip.getinfo('mmsu_toc_en_txt.json'))
#     mmsu_toc_en_txt_json = json_zip.read('mmsu_toc_en_txt.json').decode(encoding='utf-8')

# with zipfile.ZipFile(json_out_dir + 'txt/mmsu_toc_en_txt_json.zip', 'r') as json_zip:
#     mmsu_toc_en_txt_dict = json.loads(json_zip.read('mmsu_toc_en_txt.json').decode(encoding='utf-8'))

# Construct complete contents dictionsaries from zip archives
html_from_json = dict()
md_from_json = dict()
txt_from_json = dict()

for fmt in ['html', 'md', 'txt']:
    for clxn in tqdm(collection_cat.categories):
        with zipfile.ZipFile(
                f'{json_out_dir}/{fmt}/{clxn}_{fmt}_json.zip', 'r') as json_zip:
            dict_from_json = json.loads(json_zip.read(f'{clxn}_{fmt}.json')
                       .decode(encoding='utf-8'))
            eval(f'{fmt}_from_json').update(dict_from_json)
# 21/21 [00:19<00:00,  1.09it/s]
# 21/21 [00:08<00:00,  2.62it/s]
# 21/21 [00:04<00:00,  4.66it/s]
del fmt, clxn, json_zip, dict_from_json

# check mutual consistency
len(html_from_json) # 33567
len(md_from_json)   # 33567
len(txt_from_json)  # 33567
list(html_from_json) == list(md_from_json)      # True
list(html_from_json) == list(txt_from_json)     # True
list(html_from_json) == corpus_df.url.to_list() # True


#%%
# write html_json dict to permanent location
with open(f'C:/DataPulse/html-by-year/all-data-preprocessed-html.json', 'w', encoding="utf-8") as json_file:
    json.dump(html_from_json, json_file,indent=1, ensure_ascii=False)


#%%
""" convert html_from_json dictionary to dataframe """
df = pd.DataFrame.from_dict(html_from_json, orient='index').reset_index()


# %%
""" merge to add date to each record. only keep inner join. """
corpus_df_date_key=corpus_df[['url','dl_date','series', 'level', 'lang']]
corpus_df_date_key['year'] = pd.DatetimeIndex(corpus_df_date_key['dl_date']).year
corpus_df_date_key['year'] = corpus_df_date_key['year'].fillna('9999').astype(int)

html_date = pd.merge(df, corpus_df_date_key, right_on='url', left_on='index' , how='inner')
#pp.pprint(html_date)


#%%
# https://intranet.cdc.gov/connects/articles/2019/02/mmwrs-2018-top-10-reports.html
# https://intranet.cdc.gov/connects/articles/2023/03/mmwr-top-reports-of-2022.html

top10 = list()
top10 = [ # 2018
        'https://www.cdc.gov/mmwr/volumes/67/wr/mm6722a1.htm',
        'https://www.cdc.gov/mmwr/volumes/67/wr/mm6706a2.htm',
        'https://www.cdc.gov/mmwr/volumes/67/wr/mm6712a1.htm',
        'https://www.cdc.gov/mmwr/volumes/67/wr/mm6719a3.htm',
        'https://www.cdc.gov/mmwr/volumes/67/wr/mm6717e1.htm',
        'https://www.cdc.gov/mmwr/volumes/67/wr/mm6745a5.htm',
        'https://www.cdc.gov/mmwr/volumes/67/ss/ss6706a1.htm',
        'https://www.cdc.gov/mmwr/volumes/67/wr/mm675152e1.htm',
        'https://www.cdc.gov/mmwr/volumes/67/wr/mm6740a4.htm',
        'https://www.cdc.gov/mmwr/volumes/67/wr/mm6737a3.htm',

        # 2022
        'https://www.cdc.gov/mmwr/volumes/71/wr/mm7106e1.htm',
        'https://www.cdc.gov/mmwr/volumes/71/wr/mm7102e2.htm',
        'https://www.cdc.gov/mmwr/volumes/71/wr/mm7121e1.htm',
        'https://www.cdc.gov/mmwr/volumes/71/wr/mm7104e1.htm',
        'https://www.cdc.gov/mmwr/volumes/71/wr/mm7117e3.htm',
        'https://www.cdc.gov/mmwr/volumes/71/wr/mm7107e2.htm',
        'https://www.cdc.gov/mmwr/volumes/71/wr/mm7102e1.htm',
        'https://www.cdc.gov/mmwr/volumes/71/wr/mm7104e2.htm',
        'https://www.cdc.gov/mmwr/volumes/71/wr/mm7133e1.htm',
        'https://www.cdc.gov/mmwr/volumes/71/wr/mm7114e1.htm'
    ]


# %%
year = 2015
""" write JSON """
#corpus_df_date_key=corpus_df[['url','dl_date','series', 'level', 'lang']]
def get_year_data(df=pd.DataFrame(),year=year):
    df = df[df['year'].isin([year])]
    #df = df[df['url'].isin(top10)==False]
    df = df[df['lang'].isin(['en','EN'])]
    df = df[df['level'].isin(['article'])]
    yield df

years = range(2010,2015)
for y in years:
    df_dict = dict()
    for i, d in enumerate(get_year_data(html_date,y)):
        d = d.loc[:, 'index':0]
        df_dict.update(key=d['index'], value=d[0])
        pp.pprint(df_dict)
        #print(type(df_dict))

    with open(f'C:/DataPulse/data/{y}.json', 'w', encoding='utf-8') as json_file:
       json.dump(list(df_dict['value']), json_file,indent=1, ensure_ascii=False)

# %%