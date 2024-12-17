Title: Arisa: An Intelligent Chatbot

Description:
Arisa is a friendly and informative chatbot powered by machine learning techniques. She can engage in conversations, answer your questions, and provide relevant responses based on a pre-trained dataset.

Features:
1)Natural Language Processing: Utilizes tokenization, lemmatization, and TF-IDF vectorization to understand user input effectively.
2)Cosine Similarity: Identifies the most similar question from the dataset based on TF-IDF similarity for accurate responses.
3)Fuzzy Matching: When TF-IDF similarity is low, Arisa applies fuzzy matching to find the closest possible match to the user's query.
4)Greeting Recognition: Handles basic greetings like "hello" or "hi" with appropriate responses.

Installation:
Create a virtual environment (recommended for isolation).
Inside the virtual environment, install the required libraries:


Bash
pip install numpy nltk fuzzywuzzy python-Levenshtein scikit-learn
Download the NLTK data (may be required depending on your system):

Bash
python -m nltk.downloader punkt wordnet omw-1.4


Usage:
Clone this repository or download the code files.
Create a text file named datasett.txt containing your chatbot's question-answer pairs, where each line is in the format question\tanswer.

Execute the Python script:

Bash
python chatbot.py
