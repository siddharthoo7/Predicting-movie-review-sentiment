
# coding: utf-8

# In[296]:

from collections import Counter
import csv
import re
from sklearn.metrics import accuracy_score


# In[297]:

# Read in the training data.
with open("training.txt", 'r',encoding="utf8") as file:
  reviews = list(csv.reader(file))


# In[298]:

print(reviews)


# In[299]:

def get_text(reviews, score):
    ss=[]
    for r in reviews:
        if r[0][0]==str(score):
            ss.append(r)
    return ss;


# In[300]:

def count_text(text):
  # Split text into words based on whitespace.  Simple but effective.
  words = str(text).split()
  # Count up the occurence of each word.
  return Counter(words)


# In[301]:

negative_text = get_text(reviews, 0)


# In[302]:

positive_text = get_text(reviews, 1)


# In[303]:

negative_counts = count_text(negative_text)
# Generate word counts for positive tone.
positive_counts = count_text(positive_text)


# In[304]:

print(negative_counts)


# In[305]:

print(positive_counts)


# In[325]:

print("Negative text sample: {0}".format(negative_text[:3]))
print("Positive text sample: {0}".format(positive_text[:3]))


# In[326]:

from collections import Counter


# In[327]:

def get_y_count(score):
  # Compute the count of each classification occuring in the data.
    ss=[]
    for r in reviews:
        if r[0][0]==str(score):
            ss.append(r)
    return len(ss);


# In[328]:

# We need these counts to use for smoothing when computing the prediction.
positive_review_count = get_y_count(1)
negative_review_count = get_y_count(0)


# In[329]:

print(positive_review_count)
print(negative_review_count)


# In[330]:

# These are the class probabilities (we saw them in the formula as P(y)).
prob_positive = positive_review_count / len(reviews)
prob_negative = negative_review_count / len(reviews)


# In[331]:

print(prob_positive)
print(prob_negative)


# In[332]:

def make_class_prediction(text, counts, class_prob, class_count):
  prediction = 1
  text_counts = Counter(re.split("\s+", text))
  print(text_counts)
  for word in text_counts:
      # For every word in the text, we get the number of times that word occured in the reviews for a given class, add 1 to smooth the value, and divide by the total number of words in the class (plus the class_count to also smooth the denominator).
      # Smoothing ensures that we don't multiply the prediction by 0 if the word didn't exist in the training data.
      # We also smooth the denominator counts to keep things even.
      prediction *=  text_counts.get(word) * ((counts.get(word, 0) + 1) / (sum(counts.values()) + class_count))
  # Now we multiply by the probability of the class existing in the documents.
  return prediction * class_prob


# In[333]:

# As you can see, we can now generate probabilities for which class a given review is part of.
# The probabilities themselves aren't very useful -- we make our classification decision based on which value is greater.
print("Review: {0}".format(reviews[2][0]))
print("Negative prediction: {0}".format(make_class_prediction(reviews[2][0], negative_counts, prob_negative, negative_review_count)))
print("Positive prediction: {0}".format(make_class_prediction(reviews[2][0], positive_counts, prob_positive, positive_review_count)))


# In[334]:

def make_decision(text, make_class_prediction):
    # Compute the negative and positive probabilities.
    negative_prediction = make_class_prediction(text, negative_counts, prob_negative, negative_review_count)
    positive_prediction = make_class_prediction(text, positive_counts, prob_positive, positive_review_count)

    # We assign a classification based on which probability is greater.
    if negative_prediction > positive_prediction:
      return 0
    return 1


# In[335]:

with open("testdata.txt", 'r',encoding="utf8") as file:
    test = list(csv.reader(file))


# In[336]:

predictions = [make_decision(r[0], make_class_prediction) for r in test]


# In[337]:

print(predictions)


# In[338]:

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics


# In[381]:

# Generate counts from text using a vectorizer.  There are other vectorizers available, and lots of options you can set.
# This performs our step of computing word counts.
vectorizer = CountVectorizer(stop_words='english')
train_features = vectorizer.fit_transform([r[0] for r in reviews])
test_features = vectorizer.transform([r[0] for r in test])


# In[392]:

labels=[]
for r in reviews:
    labels.append(int(r[0][0]))


# In[393]:

print(labels)


# In[394]:

# Fit a naive bayes model to the training data.
# This will train the model using the word counts we computer, and the existing classifications in the training set.
nb = MultinomialNB()
nb.fit(train_features, labels)


# In[395]:

y_pred = nb.predict(test_features)
print(y_pred)
len(y_pred)


# In[396]:

print(accuracy_score(predictions, y_pred))


# In[ ]:



