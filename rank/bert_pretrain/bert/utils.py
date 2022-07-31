from tqdm import tqdm
import zhconv
import numpy as np
import random
import torch
import os
import unicodedata
import csv
import html
import re
import jionlp as jio


def set_seeds(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    
def hant_2_hans(hant_str):
    '''
    Function: 将 hant_str 由繁体转化为简体 并且将字母统一小写
    '''
    return zhconv.convert(hant_str, 'zh-hans').lower()
   
def normalize_unicode(text):
    '''
    Function: 将全角符号统一为半角
    https://stackoverflow.com/questions/2422177/python-how-can-i-replace-full-width-characters-with-half-width-characters/2422245#2422245
    '''
    return unicodedata.normalize('NFKC', text)
    

# 清洗杂项符号、表情等
# https://blog.csdn.net/m0_48742971/article/details/123117893
# 如 £ 这种符号不能被 jio.clean_text() 清除，所以加个这个方法 clean()
def clean(text:str):
    return "".join(ch for ch in text if unicodedata.category(ch)[0]!= 'S')


def clean_html_tag(text):
    html_tag = {'&#xA;': ' ', '&quot;': ' ', '&amp;': '', '&lt;': ' ', '&gt;': ' ', '&apos;': ' ',
                '&nbsp;': ' ', '&yen;': ' ', '&copy;': ' ', '&divide;': ' ','&times;': ' ',
                '&trade;': ' ', '&reg;': ' ', '&sect;': ' ', '&euro;': ' ','&pound;': ' ',
                '&cent;': ' ', '&raquo;': ' ', '&hellip;': ' ', '&lsquo;': ' ','&rsquo;': ' ',
                '&ldquo;': ' ', '&rdquo;': ' ', '&bull;': ' ', '&ndash;': ' ', '&mdash;': ' ',
                '&rsaquo;': ' ', '&middot;': ' ', '&deg;': ' ', '&ocirc;': ' ', '&sup2;': ' ',
                '&alpha;': 'a', '&agrave;': 'a', '&aacute;': 'a', '&auml;': 'a', '&aring;': 'a',
                '&ccedil;': 'c', '&egrave;': 'e', '&eacute;': 'e', '&ecirc;': 'e', '&igrave;': 'i',
                '&iacute;': 'i', '&ntilde;': 'n', '&oacute;': 'o', '&ocirc;': 'o', '&ouml;': 'o',
                '&oslash;': 'o', '&ucirc;': 'u', '&uuml;': 'u', '&radic;': ' ', '&phi;': ' phi ',
                '&shy;': '', '&omicron;': 'o', '&mu;': 'm', '&zeta;': ' ', '&beta;': 'b', '&ge;': '',
                '&kappa;': 'k', '&rho;': 'p', '&lrm;': '', '&gamma;': 'gamma'              
            }
    for k, v in html_tag.items():
        text = text.replace(k, v)
    text = html.unescape(text)
    return text


def delete_kuohao(text:str):
    text = text.replace('【', '(').replace('】', ')')
    text = text.replace('[', '(').replace(']', ')')
    return text

    
def text_preprocess(text):
    text = hant_2_hans(text)  # 繁简转换+大小写转换（统一小写）
    text = clean_html_tag(text)  # 清洗 HTML 字符实体
    text = normalize_unicode(text)  # 将全角符号统一为半角
    text = delete_kuohao(text)  # 处理文本中的括号
    text = clean(jio.clean_text(text.strip()))  # 清洗文本
    while text != text.replace("  ", " "): # 多空格转单空格
        text = text.replace("  ", " ")
    
    return text
    """
    #2.0 左右去空格
    corpus = corpus.strip()
    #2.1 繁体转简体
    corpus = zhconv.convert(corpus, 'zh-hans')
    #2.2 大小转小写
    corpus = corpus.lower()
    #2.3 多空格转单空格
    while corpus != corpus.replace("  ", " "): # 多空格转单空格
        corpus = corpus.replace("  ", " ")
    
    return corpus
    """

    
def query_vacab():
    """
    query词典
    """
    query_path = "/mnt/search/recall2/WenTianSearch/vocabulary/query.txt"
    with open(query_path, "r") as f:
        data = f.readlines()
    
    raw2new_query = {}
    for line in data:
        line = line.strip()
        line = line.split("----->")
        raw = line[0].strip()
        new = line[1].strip()
        raw2new_query[raw] = new
    return raw2new_query
    
    
def read_corpus(file_path='./data/corpus.tsv'):
    """
    获取query和doc中，id到文本的映射
    适用对象: 第一列id，第二列文本，中间用\t间隔
    """
    reader = csv.reader(open(file_path), delimiter='\t')
    total_dict = dict()
    for line in reader:
        corpus_id = int(line[0])
        corpus = line[1]
        total_dict[corpus_id] = corpus
    return total_dict
     
    
if __name__ == "__main__":
    query_vacab()