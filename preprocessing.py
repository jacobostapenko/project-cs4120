import pandas
import numpy
import sklearn
import matplotlib.pyplot as plt
from collections import Counter
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import contractions

'''
NOTES

-not sure whether you want to include all the preprocessing steps
-i figured best to drop irrelevant columns
-limited df to first 100 rows for easier dev 

'''



'''
load data and drop irrelevant columns
'''
df = pandas.read_csv('articles1.csv')
## limited df to first 100 for easier dev ##
df = df[:100]
stop_words = set(nltk.corpus.stopwords.words('english'))

print("Dataframe shape: ", df.shape)
print("stop word num: ", len(stop_words))

df = df.drop(columns=['title','date', 'year', 'month', 'url'])


'''
publisher balance and author balance. 
    -count publishers and items + graph
    -count authors and items + graph
'''
pub_counter = Counter(df['publication'])
pub_counts = [count for item, count in pub_counter.items()]
publications = [item for item, count in pub_counter.items()]

plt.figure()
plt.bar(publications, pub_counts)
#plt.show()

author_counter = Counter(df['author'])
author_counts = [count for item, count in author_counter.items()]
authors = [item for item, count in author_counter.items()]

plt.figure()
plt.bar(authors[:10], author_counts[:10])
#plt.show()


'''
tokenizing and preprocessing for each article
    -word decapitlization
    -removing stop words
    -lemmatization
    -contraction expansion
'''
lemmatizer = WordNetLemmatizer()

for article in df['content']:
    word_tokens = article.split()
    word_tokens = [word.lower() for word in word_tokens]

    processed_article = []
    for word in word_tokens:
        if word not in stop_words:
            word = lemmatizer.lemmatize(word)
            processed_article.append(contractions.fix(word))
    article = ' '.join(processed_article)
    