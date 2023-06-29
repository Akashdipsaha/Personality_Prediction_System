## Personality Prediction System via CV Analysis

This project is a Personality Prediction System that utilizes CV analysis to predict personality traits based on resumes. 
The system is implemented in Python and leverages various libraries and techniques, including natural language processing and machine learning.


### Installation
1. Clone the repository:

2. Navigate to the project directory:
   ```
   cd your-repository
   ```
3. Install the required libraries:
   ```
   pip install numpy pandas matplotlib seaborn scikit-learn nltk wordcloud
   ```
4. Download NLTK stopwords:
   ```python
   import nltk
   nltk.download('stopwords')
   ```
5. Download NLTK WordNet:
   ```python
   nltk.download('wordnet')
   ```

### Usage
1. Import the necessary libraries:
   ```python
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   import seaborn as sns
   import warnings
   warnings.filterwarnings('ignore')
   import re
   from sklearn.preprocessing import LabelEncoder
   from sklearn.model_selection import train_test_split
   from sklearn.feature_extraction.text import TfidfVectorizer
   from scipy.sparse import hstack
   from sklearn.multiclass import OneVsRestClassifier
   from sklearn.neighbors import KNeighborsClassifier
   from sklearn import metrics
   ```
2. Load the resume dataset:
   ```python
   df = pd.read_csv("path/to/UpdatedResumeDataSet.csv")
   ```
3. Explore the dataset:
   ```python
   df.head()
   ```
4. Clean the resume text:
   ```python
   def cleanResume(resumeText):
       # Remove URLs
       resumeText = re.sub('http\S+\s*', ' ', resumeText)
       # Remove RT and cc
       resumeText = re.sub('RT|cc', ' ', resumeText)
       # Remove hashtags
       resumeText = re.sub('#\S+', '', resumeText)
       # Remove mentions
       resumeText = re.sub('@\S+', '  ', resumeText)
       # Remove punctuations
       resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)
       # Remove non-ASCII characters
       resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)
       # Remove extra whitespace
       resumeText = re.sub('\s+', ' ', resumeText)
       return resumeText
   
   df['cleaned'] = df['Resume'].apply(lambda x: cleanResume(x))
   ```
5. Analyze category distribution:
   ```python
   category = df['Category'].value_counts().reset_index()
   # Display bar plot
   plt.figure(figsize=(12, 8))
   sns.barplot(x=category['Category'], y=category['index'], palette='cool')
   plt.show()
   # Display pie chart
   plt.figure(figsize=(12, 8))
   plt.pie(category['Category'], labels=category['index'], colors=sns.color_palette('cool'), autopct='%.0f%%')
   plt.title('Category Distribution')
   plt.show()
   ```
6. Vectorize the cleaned text:
   ```python
  

 corpus = " ".join(df["cleaned"])
   word_vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english', max_features=1500)
   word_vectorizer.fit(text)
   WordFeatures = word_vectorizer.transform(text)
   ```
7. Split the data into train and test sets:
   ```python
   X_train, X_test, y_train, y_test = train_test_split(WordFeatures, target, random_state=24, test_size=0.2)
   ```
8. Train the model:
   ```python
   model = OneVsRestClassifier(KNeighborsClassifier())
   model.fit(X_train, y_train)
   ```
9. Make predictions:
   ```python
   y_pred = model.predict(X_test)
   ```
10. Evaluate the model:
    ```python
    print(f'Training Accuracy: {(model.score(X_train, y_train) * 100).round(2)}%')
    print(f'Validation Accuracy: {(model.score(X_test, y_test) * 100).round(2)}%')
    print(metrics.classification_report(y_test, y_pred))
    ```
11. Generate word cloud:
    ```python
    from wordcloud import WordCloud
    
    # Create a word cloud from the lemmatized words
    res = ' '.join([i for i in lem_words if not i.isdigit()])
    plt.subplots(figsize=(16, 10))
    wordcloud = WordCloud(background_color='black', max_words=100, width=1400, height=1200).generate(res)
    plt.imshow(wordcloud)
    plt.title('Resume Text WordCloud (100 Words)')
    plt.axis('off')
    plt.show()
    ```
