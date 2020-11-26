from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle

from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.externals 
import joblib
# import sklearn.external.joblib as extjoblib
# import joblib


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	#อ่านไฟล์ จาก.csv
	df= pd.read_csv("spam.csv", encoding="latin-1")
	#drop columที่ไม่จำเป็นออก
	df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True) 
	# Features and Labels 
	#map ham => 0 และ spam => 1
	df['label'] = df['Class'].map({'ham': 0, 'spam': 1})
	df['message']=df['Text']
	#drop colum ที่ไม่จำเป็นออก
	df.drop(['Class','Text'],axis=1,inplace=True)
	X = df['message']
	y = df['label']
	
	# Extract Feature  With CountVectorizer (แยกคณสมบัติโดยการนับความถี่ของคำ)
	cv = CountVectorizer()
	X = cv.fit_transform(X) # Fit the Data
	#ขอใช้คำสั่ง train_test_split จากไลบรารี่ scikit-learn
	from sklearn.model_selection import train_test_split
	#แบ่งข้อมูลสำหรับสอน(train)ออกเป็น 70% และสำหรับการทดสอบ (test)ออกเป็น30% และกำหนดrandom_state=15 คือสุ่มทีละ 15 ตัว
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=45)
	
	
	#Naive Bayes Classifier
	from sklearn.naive_bayes import MultinomialNB #ใช้กับtext 
 	#try new model in Navie Bay	*********
	clf = MultinomialNB()
	clf.fit(X_train,y_train)
	clf.score(X_test,y_test)
	#Alternative Usage of Saved Model
	joblib.dump(clf, 'NB_spam_model.pkl')
	NB_spam_model = open('NB_spam_model.pkl','rb')
	clf = joblib.load(NB_spam_model)

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)