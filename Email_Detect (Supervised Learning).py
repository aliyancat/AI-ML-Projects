import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


df = pd.read_csv('spam_ham_dataset.csv')


x = df['text']
y = df['label']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

vectorizer = CountVectorizer()

x_train_vectors = vectorizer.fit_transform(x_train)
x_test_vectors = vectorizer.transform(x_test)

classifier = MultinomialNB()

classifier.fit(x_train_vectors,y_train)

predictions = classifier.predict(x_test_vectors)
acuracy = accuracy_score(y_test,predictions)

print(f"Accuracy: {acuracy *100:.2f}%")

test_emails = ["free money!" , "naughty calls" , "urgent meeting" , "lunch"]

test_vector = vectorizer.transform(test_emails)
test_predictions = classifier.predict(test_vector)


for email,prediciton in zip(test_emails,test_predictions):
    print(f"Email: {email} -> {prediciton}")

