import html
import emoji
import re
import contractions
from nltk.corpus import stopwords
import numpy as np
import corenlp
import os
import autocorrect
# In[92]:

class nlp_client():
    
    def __init__(self):
        directory = (os.getcwd()+r'\\corenlp')
        os.environ['CORENLP_HOME'] = directory
        prop = {'pos.model': directory + r'\gate-EN-twitter.model'}
        self.client = corenlp.CoreNLPClient(annotators="tokenize pos lemma".split(), properties=prop, output_format = 'json')
        
        #to combat connection error which occur in initial trial
        for _ in range(3):
            try:
                self.client.annotate('Hello, World')
                break
            except: continue
                
    #returns text in its lemma form and detected entities
    def annotate(self, text):
        annotation = self.client.annotate(text)
        result = annotation['sentences']
    
        c_text = []
        for sent in result:            
            text = [x['lemma'] for x in sent['tokens']]
            c_text.extend(text)

        return c_text

#splits hashtag body at capital letter and adds hashtag indicator
def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    result = " ".join(["<hashtag>"] + re.sub(r"([^A-Z])([A-Z])", r"\1 \2", hashtag_body).split())
    return result

def spell_correct(text):
    text = text.split(' ')
    c_text = []
    for word in text:
        if bool(re.match(r'\W',word)) is False:
            word = autocorrect.spell(word)
        c_text.append(word)
    c_text = ' '.join(c_text)
    return c_text

def filter_tw(tw):
    
    text = html.unescape(tw)
    text = re.sub('\n|\r', ' ', text)
    #replace RT @user with <rt2user>
    text = re.sub('RT @[a-zA-Z0-9_]+|RT@[a-zA-Z0-9_]+', '<rt_from>', text)
    #replace @mention with <user>
    text = re.sub('@[a-zA-Z0-9_]+', '<user>', text)
    #remove url
    text = re.sub(r"(http|https)?:\/\/\S+\b|www\.(\w+\.)+\S*", "", text)
    #fix hash tag
    #text = re.sub(r"#\S+", hashtag, text)
    
    # Detecting emoticon
    eyes = r"[8:=;]"
    nose = r"['`\-]?"


    text = re.sub(r"/"," / ", text)
    text = re.sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "😃", text)
    text = re.sub(r"{}{}p+".format(eyes, nose), "😂", text)
    text = re.sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "😭", text)
    text = re.sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>", text)
    text = re.sub(r"<3","<heart>", text)
    
    #change number ex.100 to <number>
    text = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>", text)
    
    #fix repettition to one  ex. aaall to all 
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    #fix can't to cannot
    text = contractions.fix(text)
    
    #multiple space into 1 space
    text = re.sub('  +',' ', text).strip()
    
    #add indicator for reply
    if re.sub(' ', '', text)[:8] == '"<user>:': text = reply(text)
    
    #fix spelling
    text = spell_correct(text)
    

    
    return text

def reply(text):
    index = text.rfind('"')
    if index == -1: return text
    rep = text[:index]
    return rep + ' <reply>' + text[index+1:]

def additional_fix(text, client, del_sw = False):
    cleaned = []
    sw = set(stopwords.words('english'))
    sw.add('hes')
    for word in text:
        #remove stop words if specified
        if del_sw is True and word in sw: continue
        
        #change emoji to representing string
        word = emoji.demojize(word)
        if word != ':' and word[0] ==':' and word[-1] == ':': pass
        #delete puctuations
        else: 
            word = re.sub(r'[^\w!?<>]','', word.lower())
            if len(word) == 0: continue


        cleaned.append(word)
    return cleaned

def remove_other(data, emoji = False, rep_ind = True):
    
    all_data = []
    exceptions = []
    if emoji is False: exceptions = ['<smile>','<lolface>', '<heart>', '<sadface>', '<neutralface>']
    if rep_ind is False: exceptions + ['<reply>']
    for text in data:
        cleaned = []
        for word in text:
            if '<' not in word: pass
            elif word not in exceptions: continue    
            cleaned.append(word)
        all_data.append(cleaned)
    return all_data

def gen_model_input(dataset):
    data, p_len = chk_max_len(dataset)
    word_id, id_word = word_id_dic(data)
    f_data = change_word_to_id(dataset, word_id, p_len)
    
    return f_data, word_id, id_word
  
def word_id_dic(data):
    word_id = {'PAD_':0}
    id_word = {0: 'PAD_'}
    for text in data:
        for word in text:
            if word not in word_id:
                new_id = len(word_id)
                word_id[word] = new_id
                id_word[new_id] = word
    
    return word_id, id_word

def change_word_to_id(dataset, word_id, pad =  None):
    f_corpus =[]
    for data in dataset:
        if pad:
            corpus = [np.array([word_id[word] for word in text]+[0]*(pad-len(text))) for text in data]
        else: corpus = [[word_id[word] for word in text] for text in data]
        f_corpus.append(corpus)
    
    return f_corpus

def chk_max_len(data):
    combined = []
    for x in data:
        combined.extend(x)
    combined_len = [len(x) for x in combined]
    return combined, max(combined_len)