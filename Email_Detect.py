import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


df = pd.read_csv('spam_ham_dataset.csv')


X = df['text']
y = df['label']

#Splitting the data, 20% for testing and 80% for learning
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)

#Convnverts the data into numbers
vectorizer = CountVectorizer()

#Counts the data to numbers and learns it as well. Fit means leanr. Transform means coount

X_train_vectors = vectorizer.fit_transform(X_train)

#Just counts thedata
X_test_vectors = vectorizer.transform(X_test)



#Creates a Naive Bayes Algo
classifier = MultinomialNB()

#Teacches the Algo based on the input data (word counts n) n the ccorrect answers
classifier.fit(X_train_vectors, y_train)

#Takes my trained classifier and feeds it the test email word counts and get back
# the predictions, one predictions for each test
predictions = classifier.predict(X_test_vectors)

#Matches the arrays , correct answers (y_test) vs predicctions
accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy:.2f}")
print(f"Got {accuracy*100:.1f}% correct!")

test_emails = [
    "Free money now !!!" ,
    "Meeting at 3 pm",
    "URGENT: Claim your prize" ,
    "Lunch party? "
]

test_vectors = vectorizer.transform(test_emails)
#Makes at test vecctor and uses that to predict our classifieer model
test_predictions = classifier.predict(test_vectors)


for email, prediction in zip(test_emails, test_predictions):
    print(f"'{email}' -> {prediction}")