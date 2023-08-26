"""
Pligarism checker using Duckduckgo or Bing search.
Bing search impelementation still needs improvement.
This can look for any exact matchs on the web for each sentece in the essay and pligarism ratio.

Author: Mohamed Mostafa
"""
import urllib
import urllib.request
import urllib.parse
import re
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
import requests

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
        try:
            result = search_source(query)
        except IndexError or AttributeError as e:
            continue
        
        # for Bing search
        # if result == None:
        #         continue
        
        sentence = re.sub(r'[^\w\s]', '', sentence)
        plain_sentence = sentence.replace(" ", "")
        
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
        

    print(f"N. Plagiarized Sentences: {plagiarized_count}")
    plagiarism_ratio = plagiarized_count / len(sentences) * 100
    print(f"Plagiarism Ratio: {plagiarism_ratio} %")
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


# duckduckgo search implementation.
# Returns the first result url.
def search_source(query):
    encoded_query = urllib.parse.quote(query)
    search_engine_url = 'http://duckduckgo.com/html/?q='
    site = urllib.request.urlopen(search_engine_url + encoded_query)
    data = site.read()
    parsed = BeautifulSoup(data, 'html.parser')
    first_link = parsed.findAll('div', {'class': re.compile('links_main*')})[0].a['href']
    decoded_link = urllib.parse.unquote(first_link.split('=')[1])
    cleaned_link =  decoded_link[:-4]
    return cleaned_link


# Bing search implementation.
# Returns the first result url.
# Needs to be improved.
# def search_source(query):
#     encoded_query = urllib.parse.quote(query)
#     response = requests.get('https://www.bing.com/search?q=' + encoded_query)
#     if response.status_code == 200:
#         soup = BeautifulSoup(response.text, "html.parser")
#         first_result_element = soup.find("h2")
#         if first_result_element:
#             link_element = first_result_element.find("a")
#             if link_element:
#                 result_link = link_element["href"]
#                 return result_link


text = """
Mobile devices cause us to neglect social interactions.
You can cut the longer texts off and only use the first 512 Tokens.
The reason I say this is because of the rise in mental health conditions including anxiety, depression, and other mental problems brought on by cyberbullying due to phone addiction. In 2020, Malaysia was rated second in Asia for youth cyberbullying. 
Kant understands the majority of people to be content to follow the guiding institutions of society, such as the Church and the Monarchy, and unable to throw off the yoke of their immaturity due to a lack of resolution to be autonomous.
The intellectual and political edifice of Christianity, seemingly impregnable in the Middle Ages, fell in turn to the assaults made on it by humanism, the Renaissance, and the Protestant Reformation.
The formative influence for the Enlightenment was not so much content as method.
a philosophical movement of the 18th century marked by a rejection of traditional social, religious, and political ideas and an emphasis on rationalism.
Similarly, early Americans had faith that a newly expanded print media would spread enlightenment. 
The idea of society as a social contract, however, contrasted sharply with the realities of actual societies.
"""


check_for_plagiarism(text)
