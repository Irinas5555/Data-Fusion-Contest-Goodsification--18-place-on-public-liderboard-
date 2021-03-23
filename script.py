import pandas as pd
import pickle

import scipy
from scipy.sparse import coo_matrix, hstack

from sklearn.svm import SVC, LinearSVC
import re
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import numpy as np

from nltk.tokenize import RegexpTokenizer
from catboost import Pool, CatBoostClassifier

# UNIQUE_NUMBER 17855781

CAT_COLS = ["receipt_dayofweek", "item_price", "item_nds_rate", "hour"]


def fts_eng(tfidf, df):
    df["hour"] = df["receipt_time"].fillna("-1:-1").apply(lambda x: x.split(":")[0])

    for i, cat_col in enumerate(CAT_COLS):
        tdf = pd.get_dummies(df[cat_col])
        tdf.columns = [cat_col + "_" + str(c) for c in tdf.columns]

        if i == 0:
            res_df = tdf
        else:
            res_df = pd.concat([res_df, tdf], axis=1)

    #     res_df["item_quantity"] = df["item_quantity"].values# / max(df["item_quantity"])

    res_df.index = range(len(res_df))

    # pickle.dump(res_df.columns, open('columns', 'wb'))
    columns = pickle.load(open('columns', 'rb'))

    for col in res_df.columns:
        if col not in columns:
            res_df[col] = 0

    res_df = res_df[columns]

    #     tfidf_train = pd.DataFrame(tfidf.transform(df.item_name).todense())
    #     res_df = pd.concat([res_df, tfidf_train], axis=1)

    res_df = scipy.sparse.csr_matrix(res_df.values)
    tfidf_train = tfidf.transform(df.item_name)
    res_df = hstack([res_df, tfidf_train])

    return res_df


def clean_text(df):
    df["item_name"] = df["item_name"].apply(lambda x: x.lower())

    for symbol in [",", "%", "(", ")", "." , "\'", "/", "-", "_", "/", "\""]:
        df["item_name"] = df["item_name"].apply(lambda x: x.replace(symbol, " "))

    df["item_name"] = df["item_name"].apply(lambda x: x.replace("ржано", "ржаной "))
    df["item_name"] = df["item_name"].apply(lambda x: x.replace("нач ", "начинкой "))
    df["item_name"] = df["item_name"].apply(lambda x: x.replace(" нач", " начинкой "))
    df["item_name"] = df["item_name"].apply(lambda x: x.replace("orbit", " орбит "))
    df["item_name"] = df["item_name"].apply(lambda x: x.replace("жев рез", " жевательная резинка "))
    df["item_name"] = df["item_name"].apply(lambda x: x.replace("ж р ", " жевательная резинка "))
    
    df["item_name"] = df["item_name"].apply(lambda x: x.replace("\"", " \" "))
    
    #v2
    df["item_name"] = df["item_name"].apply(lambda x: x.replace("0 75вин", " вино "))
    df["item_name"] = df["item_name"].apply(lambda x: x.replace(" рыбн ", " рыбная "))
    df["item_name"] = df["item_name"].apply(lambda x: x.replace("морепродуктами", " морепродукты "))
    
    #v3
    df["item_name"] = df["item_name"].apply(lambda x: x.replace("snickers", " сникерс "))
    
    #v4
    df["item_name"] = df["item_name"].apply(lambda x: x.replace(" kуриная ", " куриная "))
    df["item_name"] = df["item_name"].apply(lambda x: x.replace(" kотлета ", " котлета "))
    df["item_name"] = df["item_name"].apply(lambda x: x.replace(" kофе ", " кофе "))
    df["item_name"] = df["item_name"].apply(lambda x: x.replace("kофе ", "кофе "))
    df["item_name"] = df["item_name"].apply(lambda x: x.replace(" kофе", " кофе"))
    df["item_name"] = df["item_name"].apply(lambda x: x.replace("kофе", "кофе"))
    df["item_name"] = df["item_name"].apply(lambda x: x.replace(" kаппучино ", " каппучино "))
    df["item_name"] = df["item_name"].apply(lambda x: x.replace("сапоги", " обувь "))
    df["item_name"] = df["item_name"].apply(lambda x: x.replace("туфли", " обувь "))
    df["item_name"] = df["item_name"].apply(lambda x: x.replace("полусапоги", " обувь "))
    df["item_name"] = df["item_name"].apply(lambda x: x.replace("кеды высокие", " обувь "))
    df["item_name"] = df["item_name"].apply(lambda x: x.replace("кеды мужские", " обувь "))
    df["item_name"] = df["item_name"].apply(lambda x: x.replace("кеды женские", " обувь "))
    df["item_name"] = df["item_name"].apply(lambda x: x.replace("кеды детские", " обувь "))
    df["item_name"] = df["item_name"].apply(lambda x: x.replace("кеды муж", " обувь "))
    df["item_name"] = df["item_name"].apply(lambda x: x.replace("сандалии", " обувь "))
    df["item_name"] = df["item_name"].apply(lambda x: x.replace("пижама", " пижам "))
    df["item_name"] = df["item_name"].apply(lambda x: x.replace("пижамный", " пижам "))
    df["item_name"] = df["item_name"].apply(lambda x: x.replace("пижамные", " пижам "))
    df["item_name"] = df["item_name"].apply(lambda x: x.replace("для сна", " пижам "))
    df["item_name"] = df["item_name"].apply(lambda x: x.replace("сорочка дет", " пижам "))
    df["item_name"] = df["item_name"].apply(lambda x: x.replace("сорочка женская", " пижам "))
    df["item_name"] = df["item_name"].apply(lambda x: x.replace("сорочка жен", " пижам "))
    df["item_name"] = df["item_name"].apply(lambda x: x.replace("сорочка подростковая", " пижам ")) 
    df["item_name"] = df["item_name"].apply(lambda x: x.replace("kомплект", " комплект "))

#     df['item_name'] = df['item_name'].apply(clean_text_org)
    df['item_name'] = df['item_name'].str.replace('№', '')
    df['item_name'] = df['item_name'].str.replace('%', '')
    df['item_name'] = df['item_name'].str.replace('╣', '')
    df['item_name'] = df['item_name'].str.replace('"', '')
    df['item_name'] = df['item_name'].str.replace('<', '')
    df['item_name'] = df['item_name'].str.replace('>', '')
    df['item_name'] = df['item_name'].str.replace('#', '')
    df['item_name'] = df['item_name'].str.replace('_', '')
    df['item_name'] = df['item_name'].str.replace('+', '')
    df['item_name'] = df['item_name'].str.replace('-', '')
    df['item_name'] = df['item_name'].str.replace('&', '')
    df['item_name'] = df['item_name'].str.replace('  ', ' ')
    df['item_name'] = df['item_name'].str.replace('a', 'а')
    df['item_name'] = df['item_name'].str.replace('h', 'н')
    df['item_name'] = df['item_name'].str.replace('k', 'к')
    df['item_name'] = df['item_name'].str.replace('b', 'в')
    df['item_name'] = df['item_name'].str.replace('c', 'с')
    df['item_name'] = df['item_name'].str.replace('o', 'о')
    df['item_name'] = df['item_name'].str.replace('p', 'р')
    df['item_name'] = df['item_name'].str.replace('t', 'т')
    df['item_name'] = df['item_name'].str.replace('x', 'х')
    df['item_name'] = df['item_name'].str.replace('y', 'у')
    df['item_name'] = df['item_name'].str.replace('e', 'е')
    df['item_name'] = df['item_name'].str.replace('m', 'м')
    df['item_name'] = df['item_name'].str.replace('  ', ' ')

    df['item_name'] = df['item_name'].str.lstrip()


def create_features(data):
    data['first'] = data['item_name'].apply(lambda x: x.split()[0] if len(x.split()) > 0 else 'none')
    data['last'] = data['item_name'].apply(lambda x: x.split()[-1] if len(x.split()) > 0 else 'none')
     
    
    data['second_last_word'] = data.item_name.apply(lambda x: x.split()[-2] if len(x.split()) > 1 else 'none')
    data['second_first_word'] = data.item_name.apply(lambda x: x.split()[1] if len(x.split()) > 1 else 'none')
    
    data['first_word_len'] = data['first'].apply(len)
    data['last_word_len'] = data['last'].apply(len)
    
    data['last_first_word'] = data['last'] + data['first']
    return data


def get_sentence_vector(model, words_list):
    size = len(words_list)
    sentence_vector = np.zeros(100)
    
    for word in words_list:
        try:
            sentence_vector += model.wv[word]
        except Exception:
            size -= 1
            
    if size > 0:
        sentence_vector /= size
        
    return sentence_vector

def fill_median(df):
    df['std_item_quantity'] = df['std_item_quantity'].fillna(np.nanmedian(df['std_item_quantity']))
    df['std_item_price'] = df['std_item_price'].fillna(np.nanmedian(df['std_item_price']))
    df['std_item_nds_rate'] = df['std_item_nds_rate'].fillna(np.nanmedian(df['std_item_nds_rate']))
    df['std_hours'] = df['std_hours'].fillna(np.nanmedian(df['std_hours']))
    return df

def most_popular(arr):
  return np.argmax(np.bincount(arr))


dict_cat={0:0,  1:0,  2:0,  3:0, 
 4:1,  6:1,  7:1,  9:1,  11:1,  12:1,  13:1,
 19:2,  20:2,  24:2,  26:2,  27:2,  29:2,  30:2,  31:2,
 35:3,  36:3,  37:3,  38:3,  39:3,
 40:4,  41:4,  42:4,
 43:3,  45:3,
 46:4,  49:4,  50:4,  51:4,
 52:5,  53:5,  54:5,  55:5,  56:5,  57:5,  58:5,  60:5,  61:5,  62:5,
 66:6,  67:6,  68:6,  69:6,  70:6,  71:6,
 72:7,  73:7,  74:7,  75:7,  76:7,  77:7,  78:7,  79:7,  80:7,  81:7,  82:7,  83:7,  84:7,  85:7, 
 90:8,
 92:9,
 96:10,  97:10,  100:10,  101:10,  102:10,  103:10,
 105:11,  106:11,  107:11,  108:11,  109:11, 111:11,   114:11,  115:11,
 117:12,  118:12,  120:12,  128:12,  130:12,  133:12,
 138:13,  139:13,  140:13,
 145:14,  150:14,
 163:15,  164:15,  167:15,
 177:16,
 203:17,  204:17
   
}

def item_type(x):
  return dict_cat[x]


test = pd.read_parquet('data/task1_test_for_user.parquet')
clean_text(test)

gr_item_name = test.groupby('item_name')
test['mean_item_quantity'] = gr_item_name['item_quantity'].transform("mean")
test['sum_item_quantity'] = gr_item_name['item_quantity'].transform("sum")
test['std_item_quantity'] = gr_item_name['item_quantity'].transform("std")
test['min_item_quantity'] = gr_item_name['item_quantity'].transform("min")
test['max_item_quantity'] = gr_item_name['item_quantity'].transform("max")

test['mean_item_price'] = gr_item_name['item_price'].transform("mean")
test['sum_item_price'] = gr_item_name['item_price'].transform("sum")
test['std_item_price'] = gr_item_name['item_price'].transform("std")
test['min_item_price'] = gr_item_name['item_price'].transform("min")
test['max_item_price'] = gr_item_name['item_price'].transform("max")

test['mean_item_nds_rate'] = gr_item_name['item_nds_rate'].transform("mean")
test['sum_item_nds_rate'] = gr_item_name['item_nds_rate'].transform("sum")
test['std_item_nds_rate'] = gr_item_name['item_nds_rate'].transform("std")
test['min_item_nds_rate'] = gr_item_name['item_nds_rate'].transform("min")
test['max_item_nds_rate'] = gr_item_name['item_nds_rate'].transform("max")


test['hours'] = test['receipt_time'].apply(lambda x: int(x.split(":")[0]))

test['mean_hours'] = gr_item_name['hours'].transform("mean")
test['sum_hours'] = gr_item_name['hours'].transform("sum")
test['std_hours'] = gr_item_name['hours'].transform("std")
test['min_hours'] = gr_item_name['hours'].transform("min")
test['max_hours'] = gr_item_name['hours'].transform("max")


test['median_dayofweek'] = gr_item_name['receipt_dayofweek'].transform('median')
test['median_dayofweek'] =test['median_dayofweek'].astype(np.int8)


tokenizer = RegexpTokenizer(r'[\d.,]+|[A-Z][.A-Z]+\b\.*|\w+|\S')


test["tokens"] = test["item_name"].apply(lambda x: [e.lower() for e in tokenizer.tokenize(x)])

test = create_features(test)
test.fillna('', inplace=True)



tfidf = pickle.load(open('tfidf', 'rb'))
clf = pickle.load(open('clf_task1', 'rb'))
scl = pickle.load(open('scl', 'rb'))
model = Word2Vec.load("word2vec.model")


X_test1 = tfidf.transform(test.item_name)

X_test2 = pd.DataFrame(np.array([get_sentence_vector(model, tokens) for tokens in test["tokens"]]))
X_test2 = scl.transform(X_test2)

X_test = hstack([X_test1, scipy.sparse.csr_matrix(X_test2)])

X_meta = clf.decision_function(X_test)
X_meta_df=pd.DataFrame(X_meta, columns=clf.classes_)


test=pd.concat([test,X_meta_df],axis=1)
pred_category = clf.predict(X_test)
test['pred_category']=pred_category
test['pred_category']=test['pred_category'].apply(int)

test['item_type']=test['pred_category'].apply(item_type)
test['receipt_type']=test.groupby('receipt_id')['item_type'].transform(most_popular)



feature_names = ['receipt_id','receipt_time', 'receipt_dayofweek', 'item_name', 'item_quantity', 'item_price', 'item_nds_rate', 'mean_item_quantity', 'sum_item_quantity', 'std_item_quantity', 
                 'min_item_quantity', 'max_item_quantity', 'mean_item_price', 'sum_item_price', 'std_item_price', 'min_item_price', 'max_item_price', 'mean_item_nds_rate', 'sum_item_nds_rate', 
                 'std_item_nds_rate', 'min_item_nds_rate', 'max_item_nds_rate', 'hours', 'mean_hours', 'sum_hours', 'std_hours', 'min_hours', 'max_hours', 'median_dayofweek',    
                0,  1,  2,  3,  4,  6,  7,  9,  11,  12,  13,  19,  20,  24,  26,  27,  29,  30,  31,  35,  36,  37,  38,  39,  40,  41,  42,  43,  45,  46,  49,  50,  51,  52,  53,  54,  55,
                56,  57,  58,  60,  61,  62,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  90,  92,  96,  97,  100,  101,  102,  103,  105,
                106,  107,  108,  109,  111,  114,  115,  117,  118,  120,  128,  130,  133,  138,  139,  140,  145,  150,  163,  164,  167,  177,  203,  
                'pred_category', 'first','last', 

                
                'first_word_len', 'last_word_len',
                 'second_last_word', 'second_first_word',
                 
                 'last_first_word',
                 'receipt_type'
                ] 

cat_features = ['receipt_id', 'receipt_time','receipt_dayofweek', 'hours', 'min_hours', 'max_hours', 'median_dayofweek', 'first', 'last', 'second_last_word', 'second_first_word',
                 'last_first_word','pred_category', 'receipt_type'
                ]

text_features = ['item_name']

                                 
model_catboost = CatBoostClassifier()

model_catboost.load_model("model_catboost.cbm")

pred=model_catboost.predict(test[feature_names])


res = pd.DataFrame(pred, columns=['pred'])
res['id'] = test['id']

res[['id', 'pred']].to_csv('answers.csv', index=None)
