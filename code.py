import numpy as np
import nltk
import string
import random
f=open('/content/datasett.txt','r',errors='ignore')
raw_doc=f.read()
raw_doc=raw_doc.lower()
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
sentence_tokens=nltk.sent_tokenize(raw_doc)
word_tokens=nltk.word_tokenize(raw_doc)
lemmer=nltk.stem.WordNetLemmatizer()
def Lemtokens(tokens):
  return [lemmer.lemmatize(token) for token in tokens]
remove_punc_dict=dict((ord(punct),None)for punct in string.punctuation)
def LemNormalize(text):
  return Lemtokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))
greet_inputs=('hello','hi','whassup','how are you?','hey')
greet_responses=('hi','hey','hey there!','hey,i am glad you are talking to me')
def greet(sentence):
  for word in sentence.split():
    if word.lower() in greet_inputs:
      return random.choice(greet_responses)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def response(user_response):
  robol_response=''
  TfidfVec=TfidfVectorizer(tokenizer=LemNormalize,stop_words='english')
  tfidf=TfidfVec.fit_transform(sentence_tokens)
  vals=cosine_similarity(tfidf[-1],tfidf)
  idx=vals.argsort()[0][-2]
  flat=vals.flatten()
  flat.sort()
  req_tfidf=flat[-2]
  if(req_tfidf==0):
    robol_response=robol_response + "I am sorry.Unable to understand you!"
    return robol_response
  else:
    robol_response=robol_response+sentence_tokens[idx]
    return robol_response 
flag=True
print('Hello! My Name is arisa.It is nice to connect with you')
while (flag== True):
  user_response=input()
  user_response=user_response.lower()
  if(user_response !='bye'):
    if(user_response=='thank you' or user_response=='thanks'):
      flag=False
      print('You are welcome.')
    else:
      if(greet(user_response)!=None):
        print('Arisa:'+ greet(user_response))
      else:
        sentence_tokens.append(user_response)
        word_tokens=word_tokens + nltk.word_tokenize(user_response)
        final_words=list(set(word_tokens))
        print('Arisa:', end='')
        print(response(user_response))
        sentence_tokens.remove(user_response)
  else:
    flag=False
    print('Arisa:','Goodbye!')
