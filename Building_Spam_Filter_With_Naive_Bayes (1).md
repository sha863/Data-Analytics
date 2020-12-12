

# Building a Spam Filter with Naive Bayes
In this project, we're going to build a spam filter for SMS messages using the multinomial Naive Bayes algorithm. Our goal is to write a program that classifies new messages with an accuracy greater than 80% — so we expect that more than 80% of the new messages will be classified correctly as spam or ham (non-spam).

To train the algorithm, we'll use a dataset of 5,572 SMS messages that are already classified by humans. The dataset was put together by Tiago A. Almeida and José María Gómez Hidalgo, and it can be downloaded from the [The UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection). The data collection process is described in more details on [this page](http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/#composition), where you can also find some of the papers authored by Tiago A. Almeida and José María Gómez Hidalgo.

# Exploring the Dataset
We'll now start by reading in the dataset


```python
import pandas as pd

sms_spam = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['Label', 'SMS'])# To name the column name

print(sms_spam.shape)
sms_spam.head()
```

    (5572, 2)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>SMS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Ok lar... Joking wif u oni...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>U dun say so early hor... U c already then say...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
    </tr>
  </tbody>
</table>
</div>




```python
sms_spam['Label'].value_counts(normalize=True)
```




    ham     0.865937
    spam    0.134063
    Name: Label, dtype: float64




 We see that about 87% of the messages are ham, and the remaining 13% are spam. This sample looks representative, since in practice most messages that people receive are ham.




```python
# Randomize the dataset
data_randomized = sms_spam.sample(frac=1, random_state=1)

# Calculate index for split
training_test_index = round(len(data_randomized) * 0.8)

# Training/Test split
training_set = data_randomized[:training_test_index].reset_index(drop=True)
test_set = data_randomized[training_test_index:].reset_index(drop=True)

print(training_set.shape)
print(test_set.shape)
```

    (4458, 2)
    (1114, 2)


We'll now analyze the percentage of spam and ham messages in the training and test sets. We expect the percentages to be close to what we have in the full dataset, where about 87% of the messages are ham, and the remaining 13% are spam.


```python
training_set['Label'].value_counts(normalize=True)

```




    ham     0.86541
    spam    0.13459
    Name: Label, dtype: float64




```python
test_set['Label'].value_counts(normalize=True)

```




    ham     0.868043
    spam    0.131957
    Name: Label, dtype: float64




The results look good! We'll now move on to cleaning the dataset.

Data Cleaning
To calculate all the probabilities required by the algorithm, we'll first need to perform a bit of data cleaning to bring the data in a format that will allow us to extract easily all the information we need.

#### Letter Case and Punctuation
We'll begin with removing all the punctuation and bringing every letter to lower case.



```python
# Before cleaning
training_set.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>SMS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Yep, by the pretty sculpture</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Yes, princess. Are you going to make me moan?</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ham</td>
      <td>Welp apparently he retired</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>Havent.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>I forgot 2 ask ü all smth.. There's a card on ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
training_set['SMS'] = training_set['SMS'].str.replace('\W', ' ')
training_set['SMS'] = training_set['SMS'].str.lower()
training_set.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>SMS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>yep  by the pretty sculpture</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>yes  princess  are you going to make me moan</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ham</td>
      <td>welp apparently he retired</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>havent</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>i forgot 2 ask ü all smth   there s a card on ...</td>
    </tr>
  </tbody>
</table>
</div>



# Creating the Vocabulary
Let's now move to creating the vocabulary, which in this context means a list with all the unique words in our training set.

Create a vocabulary for the messages in the training set. The vocabulary should be a Python list containing all the unique words across all messages, where each word is represented as a string.
Begin by transforming each message from the SMS column into a list by splitting the string at the space character — use the Series.str.split() method.
Initiate an empty list named vocabulary.
Iterate over the the SMS column (each message in this column should be a list of strings by the time you start this loop).
Using a nested loop, iterate each message in the SMS column (each message should be a list of strings) and append each string (word) to the vocabulary list.
Transform the vocabulary list into a set using the set() function. This will remove the duplicates from the vocabulary list.
Transform the vocabulary set back into a list using the list() function.


```python
training_set['SMS'] = training_set['SMS'].str.split()

vocabulary = []
for sms in training_set['SMS']:
    for word in sms:
        vocabulary.append(word)
        
vocabulary = list(set(vocabulary))
```


```python
len(vocabulary)
```




    7783




The Final Training Set
We're now going to use the vocabulary we just created to make the data transformation we want.

To create the dictionary we need for our training set, we can use the code below, where:

* We start by initializing a dictionary named word_counts_per_sms, where each key is a unique word (a string) from the vocabulary, and each value is a list of the length of training set, where each element in the list is a 0.

* The code [0] * 5 outputs [0, 0, 0, 0, 0]. So the code [0] * len(training_set['SMS']) outputs a list of the length of training_set['SMS'], where each element in the list will be a 0.

* We loop over training_set['SMS'] using at the same time the enumerate() function to get both the index and the SMS message (index and sms).

* Using a nested loop, we loop over sms (where sms is a list of strings, where each string represents a word in a message).
We incremenent word_counts_per_sms[word][index] by 1.

  


```python
word_counts_per_sms = {unique_word: [0] * len(training_set['SMS']) for unique_word in vocabulary}

for index, sms in enumerate(training_set['SMS']):
    for word in sms:
        word_counts_per_sms[word][index] += 1
```


```python
word_counts = pd.DataFrame(word_counts_per_sms)
word_counts.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>00</th>
      <th>000</th>
      <th>000pes</th>
      <th>008704050406</th>
      <th>0089</th>
      <th>01223585334</th>
      <th>02</th>
      <th>0207</th>
      <th>02072069400</th>
      <th>...</th>
      <th>zindgi</th>
      <th>zoe</th>
      <th>zogtorius</th>
      <th>zouk</th>
      <th>zyada</th>
      <th>é</th>
      <th>ú1</th>
      <th>ü</th>
      <th>〨ud</th>
      <th>鈥</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 7783 columns</p>
</div>




```python
training_set_clean = pd.concat([training_set, word_counts], axis=1)
training_set_clean.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>SMS</th>
      <th>0</th>
      <th>00</th>
      <th>000</th>
      <th>000pes</th>
      <th>008704050406</th>
      <th>0089</th>
      <th>01223585334</th>
      <th>02</th>
      <th>...</th>
      <th>zindgi</th>
      <th>zoe</th>
      <th>zogtorius</th>
      <th>zouk</th>
      <th>zyada</th>
      <th>é</th>
      <th>ú1</th>
      <th>ü</th>
      <th>〨ud</th>
      <th>鈥</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>[yep, by, the, pretty, sculpture]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>[yes, princess, are, you, going, to, make, me,...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ham</td>
      <td>[welp, apparently, he, retired]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>[havent]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>[i, forgot, 2, ask, ü, all, smth, there, s, a,...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 7785 columns</p>
</div>



Now that we're done with data cleaning and have a training set to work with, we can begin creating the spam filter. Recall that the Naive Bayes algorithm will need to know the probability values of the two equations below to be able to classify new messages:

\begin{equation}
P(Spam | w_1,w_2, ..., w_n) \propto P(Spam) \cdot \prod_{i=1}^{n}P(w_i|Spam) \\
P(Ham | w_1,w_2, ..., w_n) \propto P(Ham) \cdot \prod_{i=1}^{n}P(w_i|Ham)
\end{equation}

Also, to calculate P(wi|Spam) and P(wi|Ham) inside the formulas above, recall that we need to use these equations:

\begin{equation}
P(w_i|Spam) = \frac{N_{w_i|Spam} + \alpha}{N_{Spam} + \alpha \cdot N_{Vocabulary}} \\
P(w_i|Ham) = \frac{N_{w_i|Ham} + \alpha}{N_{Ham} + \alpha \cdot N_{Vocabulary}}
\end{equation}

Some of the terms in the four equations above will have the same value for every new message. As a start, let's first calculate:

* P(Spam) and P(Ham)
* NSpam, NHam, NVocabulary


Recall from the previous mission that:

* NSpam is equal to the number of words in all the spam messages — it's not equal to the number of spam messages, and it's not equal to the total number of unique words in spam messages.

* NHam is equal to the number of words in all the non-spam messages — it's not equal to the number of non-spam messages, and it's not equal to the total number of unique words in non-spam messages.

We'll also use Laplace smoothing and set alpha =1



```python

# Isolating spam and ham messages first
spam_messages = training_set_clean[training_set_clean['Label'] == 'spam']
ham_messages = training_set_clean[training_set_clean['Label'] == 'ham']

# P(Spam) and P(Ham)
p_spam = len(spam_messages) / len(training_set_clean)
p_ham = len(ham_messages) / len(training_set_clean)


# N_Spam
n_words_per_spam_message = spam_messages['SMS'].apply(len)
n_spam = n_words_per_spam_message.sum()

# N_Ham
n_words_per_ham_message = ham_messages['SMS'].apply(len)
n_ham = n_words_per_ham_message.sum()

# N_Vocabulary
n_vocabulary = len(vocabulary)

# Laplace smoothing
alpha = 1

```

We managed to calculate a few terms for our equations:

* P(Spam) and P(Ham)
* NSpam, NHam, NVocabulary

As we've already mentioned, all these terms will have constant values in our equations for every new message (regardless of the message or each individual word in the message).

However, P(wi|Spam) and P(wi|Ham) will vary depending on the individual words. For instance, P("secret"|Spam) will have a certain probability value, while P("cousin"|Spam) or P("lovely"|Spam) will most likely have other values.

Although both P(wi|Spam) and P(wi|Ham) vary depending on the word, the probability for each individual word is constant for every new message.

For instance, let's say we receive two new messages:

* "secret code"
* "secret party 2night"

We'll need to calculate P("secret"|Spam) for both these messages, and we can use the training set to get the values we need to find a result for the equation below:

\begin{equation}
P(\text{"secret"}|Spam) = \frac{N_{"secret"|Spam} + \alpha}{N_{Spam} + \alpha \cdot N_{Vocabulary}}
\end{equation}

The steps we take to calculate P("secret"|Spam) will be identical for both of our new messages above, or for any other new message that contains the word "secret". The key detail here is that calculating P("secret"|Spam) only depends on the training set, and as long as we don't make changes to the training set, P("secret"|Spam) stays constant. The same reasoning also applies to P("secret"|Ham).

This means that we can use our training set to calculate the probability for each word in our vocabulary. If our vocabulary contained only the words "lost", "navigate", and "sea", then we'd need to calculate six probabilities:

* P("lost"|Spam) and P("lost"|Ham)
* P("navigate"|Spam) and P("navigate"|Ham)
* P("sea"|Spam) and P("sea"|Ham)

We have 7,783 words in our vocabulary, which means we'll need to calculate a total of 15,566 probabilities. For each word, we need to calculate both P(wi|Spam) and P(wi|Ham).

In more technical language, the probability values that P(wi|Spam) and P(wi|Ham) will take are called parameters.

The fact that we calculate so many values before even beginning the classification of new messages makes the Naive Bayes algorithm very fast (especially compared to other algorithms). When a new message comes in, most of the needed computations are already done, which enables the algorithm to almost instantly classify the new message.

If we didn't calculate all these values beforehand, then all these calculations would need to be done every time a new message comes in. Imagine the algorithm will be used to classify 1,000,000 new messages. Why repeat all these calculations 1,000,000 times when we could just do them once at the beginning?

Let's now calculate all the parameters using the equations below:

\begin{equation}
P(w_i|Spam) = \frac{N_{w_i|Spam} + \alpha}{N_{Spam} + \alpha \cdot N_{Vocabulary}} \\
P(w_i|Ham) = \frac{N_{w_i|Ham} + \alpha}{N_{Ham} + \alpha \cdot N_{Vocabulary}}
\end{equation}

Recall that P(wi|Spam) and P(wi|Ham) are key parts of the equations that we need to classify the new messages:

\begin{equation}
P(Spam | w_1,w_2, ..., w_n) \propto P(Spam) \cdot \prod_{i=1}^{n}P(w_i|Spam) \\
P(Ham | w_1,w_2, ..., w_n) \propto P(Ham) \cdot \prod_{i=1}^{n}P(w_i|Ham)
\end{equation}

# Calculating Parameters
Now that we have the constant terms calculated above, we can move on with calculating the parameters $P(w_i|Spam)$ and $P(w_i|Ham)$. Each parameter will thus be a conditional probability value associated with each word in the vocabulary.

The parameters are calculated using the formulas:

$$
P(w_i|Spam) = \frac{N_{w_i|Spam} + \alpha}{N_{Spam} + \alpha \cdot N_{Vocabulary}}
$$$$
P(w_i|Ham) = \frac{N_{w_i|Ham} + \alpha}{N_{Ham} + \alpha \cdot N_{Vocabulary}}
$$


```python
# Initiate parameters
parameters_spam = {unique_word:0 for unique_word in vocabulary}
parameters_ham = {unique_word:0 for unique_word in vocabulary}

# Calculate parameters
for word in vocabulary:
    n_word_given_spam = spam_messages[word].sum()   # spam_messages already defined in a cell above
    p_word_given_spam = (n_word_given_spam + alpha) / (n_spam + alpha*n_vocabulary)
    parameters_spam[word] = p_word_given_spam
    
    n_word_given_ham = ham_messages[word].sum()   # ham_messages already defined in a cell above
    p_word_given_ham = (n_word_given_ham + alpha) / (n_ham + alpha*n_vocabulary)
    parameters_ham[word] = p_word_given_ham
```

Classifying A New Message
Now that we have all our parameters calculated, we can start creating the spam filter. The spam filter can be understood as a function that:

* Takes in as input a new message (w1, w2, ..., wn).
* Calculates P(Spam|w1, w2, ..., wn) and P(Ham|w1, w2, ..., wn).
* Compares the values of P(Spam|w1, w2, ..., wn) and P(Ham|w1, w2, ..., wn), and:
    * If P(Ham|w1, w2, ..., wn) > P(Spam|w1, w2, ..., wn), then the message is classified as ham.
    * If P(Ham|w1, w2, ..., wn) < P(Spam|w1, w2, ..., wn), then the message is classified as spam.
    * If P(Ham|w1, w2, ..., wn) = P(Spam|w1, w2, ..., wn), then the algorithm may request human help.


```python

import re

def classify(message):
    '''
    message: a string
    '''
    
    message = re.sub('\W', ' ', message)
    message = message.lower().split()
    
    p_spam_given_message = p_spam
    p_ham_given_message = p_ham

    for word in message:
        if word in parameters_spam:
            p_spam_given_message *= parameters_spam[word]
            
        if word in parameters_ham:
            p_ham_given_message *= parameters_ham[word]
            
    print('P(Spam|message):', p_spam_given_message)
    print('P(Ham|message):', p_ham_given_message)
    
    if p_ham_given_message > p_spam_given_message:
        print('Label: Ham')
    elif p_ham_given_message < p_spam_given_message:
        print('Label: Spam')
    else:
        print('Equal proabilities, have a human classify this!')
```


```python
classify('WINNER!! This is the secret code to unlock the money: C3421.')
```

    P(Spam|message): 1.3481290211300841e-25
    P(Ham|message): 1.9368049028589875e-27
    Label: Spam



```python
classify("Sounds good, Tom, then see u there")
```

    P(Spam|message): 2.4372375665888117e-25
    P(Ham|message): 3.687530435009238e-21
    Label: Ham



# Measuring the Spam Filter's Accuracy
The two results above look promising, but let's see how well the filter does on our test set, which has 1,114 messages.

We'll start by writing a function that returns classification labels instead of printing them.


```python
def classify_test_set(message):    
    '''
    message: a string
    '''
    
    message = re.sub('\W', ' ', message)
    message = message.lower().split()
    
    p_spam_given_message = p_spam
    p_ham_given_message = p_ham

    for word in message:
        if word in parameters_spam:
            p_spam_given_message *= parameters_spam[word]
            
        if word in parameters_ham:
            p_ham_given_message *= parameters_ham[word]
    
    if p_ham_given_message > p_spam_given_message:
        return 'ham'
    elif p_spam_given_message > p_ham_given_message:
        return 'spam'
    else:
        return 'needs human classification'
```


Now that we have a function that returns labels instead of printing them, we can use it to create a new column in our test set.


```python
test_set['predicted'] = test_set['SMS'].apply(classify_test_set)
test_set.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>SMS</th>
      <th>predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Later i guess. I needa do mcat study too.</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>But i haf enuff space got like 4 mb...</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Had your mobile 10 mths? Update to latest Oran...</td>
      <td>spam</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>All sounds good. Fingers . Makes it difficult ...</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>All done, all handed in. Don't know if mega sh...</td>
      <td>ham</td>
    </tr>
  </tbody>
</table>
</div>




Now, we'll write a function to measure the accuracy of our spam filter to find out how well our spam filter does.


```python
correct = 0
total = test_set.shape[0]
    
for row in test_set.iterrows():
    row = row[1]
    if row['Label'] == row['predicted']:
        correct += 1
        
print('Correct:', correct)
print('Incorrect:', total - correct)
print('Accuracy:', correct/total)
```

    Correct: 1100
    Incorrect: 14
    Accuracy: 0.9874326750448833


The accuracy is close to 98.74%, which is really good. Our spam filter looked at 1,114 messages that it hasn't seen in training, and classified 1,100 correctly.

# Next Steps
In this project, we managed to build a spam filter for SMS messages using the multinomial Naive Bayes algorithm. The filter had an accuracy of 98.74% on the test set we used, which is a pretty good result. Our initial goal was an accuracy of over 80%, and we managed to do way better than that.

Next steps include:

* Analyze the 14 messages that were classified incorrectly and try to figure out why the algorithm classified them incorrectly
* Make the filtering process more complex by making the algorithm sensitive to letter case
