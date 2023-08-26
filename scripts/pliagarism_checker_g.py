"""
Pligarism checker using googlesearch.
Google Search API is needed for large-scale implementation.
This can look for any exact matchs on the web for each sentece in the essay and pligarism ratio.

Author: Mohamed Mostafa
"""
from googlesearch import search
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
import requests
from bs4 import BeautifulSoup
import re

def preprocess_essay(essay):
    essay_sentences = []
    essay_pre = essay.lower().replace(":", ". ").replace("—", ". ").replace(";", ". ").replace('\n', ' ').strip()
    essay_tokens = sent_tokenize(essay_pre)
    for i in range (len(essay_tokens)):
        if len(essay_tokens[i]) > 33:
            essay_sentences.append(essay_tokens[i])
    return essay_sentences

def check_for_plagiarism(text):
    print("Original Text :")
    print(text.replace('\n', ' ').replace('\r', ''))
    print("==========================================================")
    sentences = preprocess_essay(text)
    plagiarized_sentences = []
    sources = []
    plagiarized_count = 0
    print("Checking is in progress...")

    for sentence in sentences:
        query = f'"{sentence}"'
        search_results = search(query, num_results=1)   # Change for Api in case of large-scale deployment.
        sentence = re.sub(r'[^\w\s]', '', sentence)
        plain_sentence = sentence.replace(" ", "")
        
        for result in search_results:
            try:
                site_text = get_visible_text(result)
                site = preprocess_essay(site_text)
            except TypeError as e:
                continue
            
            for web_sentence in site:
                web_sentence = re.sub(r'[^\w\s]', '', web_sentence)
                plain_web_sentence = web_sentence.replace(" ", "")
                if plain_web_sentence == plain_sentence:
                    plagiarized_count += 1
                    plagiarized_sentences.append(sentence)
                    print("Found: " + web_sentence)
                    sources.append(result)
                    print("Source: " + result)
            break
        

    print(f"Plagiarized Sentences N.: {plagiarized_count}")
    plagiarism_ratio = plagiarized_count / len(sentences)
    print(f"Plagiarism Ratio: {plagiarism_ratio:.2f}")
    print("Plagiarized Sentences: ")
    print(plagiarized_sentences)
    print("Sources: ")
    print(sources)

    return plagiarism_ratio


def get_visible_text(url):
    if url != None:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        text_elements = soup.get_text().split()
        visible_text = ' '.join(element.strip() for element in text_elements if element.strip())
        return visible_text


text = """
Mobile devices cause us to neglect social interactions.
You can cut the longer texts off and only use the first 512 Tokens.
The reason I say this is because of the rise in mental health conditions including anxiety, depression, and other mental problems brought on by cyberbullying due to phone addiction. In 2020, Malaysia was rated second in Asia for youth cyberbullying. 
Kant understands the majority of people to be content to follow the guiding institutions of society, such as the Church and the Monarchy, and unable to throw off the yoke of their immaturity due to a lack of resolution to be autonomous.
The intellectual and political edifice of Christianity, seemingly impregnable in the Middle Ages, fell in turn to the assaults made on it by humanism, the Renaissance, and the Protestant Reformation.
The formative influence for the Enlightenment was not so much content as method.
a philosophical movement of the 18th century marked by a rejection of traditional social, religious, and political ideas and an emphasis on rationalism
Similarly, early Americans had faith that a newly expanded print media would spread enlightenment.
The idea of society as a social contract, however, contrasted sharply with the realities of actual societies.
"""



check_for_plagiarism(text)

