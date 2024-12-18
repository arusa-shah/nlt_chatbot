import json
import random
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('wordnet')

lemmer = nltk.stem.WordNetLemmatizer()

# Load intents.json
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Extract patterns and responses
patterns = []
responses = []

for intent in intents:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        responses.append(random.choice(intent['responses']))

# Preprocessing function
def LemNormalize(text):
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    return ' '.join([lemmer.lemmatize(token) for token in nltk.word_tokenize(
        text.lower().translate(remove_punct_dict))])

# Function to generate a response
def get_response(user_response):
    user_response_processed = LemNormalize(user_response)
    temp_patterns = patterns + [user_response_processed]
    
    # TF-IDF Vectorization
    TfidfVec = TfidfVectorizer()
    tfidf = TfidfVec.fit_transform(temp_patterns)
    cosine_similarities = cosine_similarity(tfidf[-1], tfidf[:-1])
    
    # Find the best match
    idx = cosine_similarities.argsort()[0][-1]
    flat = cosine_similarities.flatten()
    flat.sort()
    highest_similarity = flat[-1]
    
    threshold = 0.3  # Adjust based on desired sensitivity
    if highest_similarity >= threshold:
        return responses[idx]
    else:
        return "I'm sorry, I couldn't find a relevant response."

# Chatbot loop
def chatbot():
    print("Arisa: Hello! My name is Arisa. It's nice to connect with you.")
    flag = True
    while flag:
        user_input = input().lower()
        if user_input in ['bye', 'exit', 'quit']:
            flag = False
            print("Arisa: Goodbye! Have a great day.")
        elif user_input in ['thanks', 'thank you']:
            print("Arisa: You're welcome!")
        else:
            response = get_response(user_input)
            print(f"Arisa: {response}")

# Run the chatbot
chatbot()

!pip install fuzzywuzzy
!pip install python-Levenshtein

# Import required libraries
import numpy as np
import nltk
import string
import random
from fuzzywuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Step 1: Load the Dataset and Split into Questions and Answers
# Replace 'datasett.txt' with the path to your dataset file if different
with open('datasett.txt', 'r', errors='ignore') as f:
    raw_doc = f.read().lower()

# Split the raw document into lines
lines = raw_doc.splitlines()

# Split each line into question-answer pairs
conversation_pairs = [line.split('\t') for line in lines if '\t' in line]

# Create a question-answer dictionary
qa_pairs = {pair[0]: pair[1] for pair in conversation_pairs}
questions = list(qa_pairs.keys())

# Step 2: Preprocessing with Lemmatization
lemmer = nltk.stem.WordNetLemmatizer()

def LemNormalize(text):
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    return ' '.join([lemmer.lemmatize(token) for token in nltk.word_tokenize(
        text.lower().translate(remove_punct_dict))])

# Greetings Inputs and Responses
greet_inputs = ('hello', 'hi', 'whassup', 'how are you?', 'hey')
greet_responses = ('hi', 'hey', 'hey there!', 'hello, I’m glad you’re here!')

def greet(sentence):
    for word in sentence.split():
        if word.lower() in greet_inputs:
            return random.choice(greet_responses)
    return None

# Step 3: Response Matching Function
def get_response(user_response):
    # Preprocess the user response
    user_response_processed = LemNormalize(user_response)
    
    # Append the user response to the list of questions
    temp_questions = questions + [user_response_processed]
    
    # Vectorize the questions using TF-IDF
    TfidfVec = TfidfVectorizer(stop_words='english')
    tfidf = TfidfVec.fit_transform(temp_questions)
    
    # Compute cosine similarity between the user response and all questions
    vals = cosine_similarity(tfidf[-1], tfidf[:-1])
    idx = vals.argsort()[0][-1]  # Index of the most similar question
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-1]
    
    # Threshold for selecting a response
    threshold = 0.3
    
    if req_tfidf >= threshold:
        return qa_pairs[questions[idx]]
    else:
        # Use fuzzy matching if TF-IDF similarity is low
        closest_match, score = process.extractOne(user_response_processed, questions)
        if score > 60:  # Fuzzy matching threshold
            return qa_pairs[closest_match]
        else:
            return "I'm sorry, I couldn't find a relevant response."

# Step 4: Chatbot Interaction Loop
def chatbot():
    print("Arisa: Hello! My name is Arisa. It's nice to connect with you.")
    flag = True
    while flag:
        user_response = input().lower()
        if user_response.strip() != '':
            if user_response in ['bye', 'exit', 'quit']:
                flag = False
                print("Arisa: Goodbye! Have a great day.")
            elif user_response in ['thanks', 'thank you']:
                print("Arisa: You're welcome!")
            elif greet(user_response) is not None:
                print("Arisa:", greet(user_response))
            else:
                response = get_response(user_response)
                print("Arisa:", response)
        else:
            print("Arisa: Please say something.")

# Run the chatbot
chatbot()
