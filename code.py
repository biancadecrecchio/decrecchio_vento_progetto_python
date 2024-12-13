#!/usr/bin/env python
# coding: utf-8

# INSTALLAZIONE PACCHETTI
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import WordNetLemmatizer
from collections import Counter
from nltk.util import ngrams
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from matplotlib_venn import venn2
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import csv
import nltk
import unittest

# ANALISI DEL DATASET
# Importazione dataset

data = pd.read_csv("combined_data.csv") 

# Pre-processing del testo per permettere l'analisi delle parole

# Risorse per tokenizzazione, rimozione stopwords e lemmatizzazione
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Funzione per processare il testo
def preprocess_text(text):
    # Rimozione punteggiatura
    text = str(text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = text.lower()

    # Tokenizzazione
    words = word_tokenize(text)

    # Rimozione stopwords e lemmatizzazione
    words = [lemmatizer.lemmatize(word)
             for word in words if word not in stop_words]

    return ' '.join(words)


# Applicazione del preprocessing agli enunciati
data['processed_text'] = data['statement'].apply(preprocess_text)

# Separazione degli enunciati in base allo status
statuses = data['status'].unique() 

texts_by_status = {status: data[data['status'] == status]['processed_text'] for status in statuses}

# Funzione per calcolare le parole più frequenti
def get_most_freq_words(texts, n=100):
    all_words = ' '.join(texts).split()
    word_freq = Counter(all_words)
    return word_freq.most_common(n)

# Salvataggio delle parole più frequenti
with open('frequent_words_outputs.txt', 'w') as words_file:
    for status, texts in texts_by_status.items():
        top_words = get_most_freq_words(texts)
        words_file.write(f"Parole più frequenti per {status}:\n{top_words}\n\n")
        
# Funzione per calcolare i bigrammi più frequenti
def get_most_freq_bigrams(texts, n=100):
    bigram_freq = Counter()
    for text in texts:
        words = text.split()
        bigrams = ngrams(words, 2)
        bigram_freq.update(bigrams)
    return bigram_freq.most_common(n)

# Salvataggio dei bigrammi più frequenti
with open('frequent_bigrams_outputs.txt', 'w') as bigrams_file:
    for status, texts in texts_by_status.items():
        top_bigrams = get_most_freq_bigrams(texts)
        bigrams_file.write(f"Bigrammi più frequenti per {status}:\n{top_bigrams}\n\n")
        
# Sentiment Analysis con VADER
nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

# Funzione per calcolare il punteggio di polarità
def get_sentiment_score(text):
    score = sia.polarity_scores(text)
    return score['compound']

# Applicazione della sentiment analysis al testo processato
data['sentiment_score'] = data['processed_text'].apply(get_sentiment_score)

# Classificazione del sentiment basata sul punteggio di polarità
data['sentiment_category'] = data['sentiment_score'].apply(
    lambda x: 'Positive' if x > 0.05 else (
        'Negative' if x < -0.05 else 'Neutral'))

# Salvataggio in file CSV
data[['processed_text', 'sentiment_score', 'sentiment_category']].to_csv(
    'sentiment_analysis_results.csv', index=False, sep=';')

# Calcolo delle frequenze relative del sentiment per ogni status
total_counts = data.groupby('status').size().reset_index(name='total_counts')

sentiment_counts_by_status = data.groupby(
    ['status', 'sentiment_category']).size().reset_index(name='counts')
sentiment_counts_by_status = sentiment_counts_by_status.merge(
    total_counts, on='status')

sentiment_counts_by_status['proportion'] = sentiment_counts_by_status['counts'] / \
    sentiment_counts_by_status['total_counts']

# Calcolo delle frequenze relative del sentiment per popolazione di pazienti e gruppo di controllo
data['group'] = data['status'].apply(
    lambda x: 'C.G.' if x == 'Normal' else 'P.P.')

grouped_data = data.groupby(
    ['group', 'sentiment_category']).size().reset_index(name='counts')

group_totals = grouped_data.groupby('group')['counts'].transform('sum')
grouped_data['relative_freq'] = grouped_data['counts'] / group_totals

# Salvataggio percentuali in file .txt
with open('sentiment_distributions.txt', 'w') as file:
    file.write("Relative distribution of sentiment by Status:\n")
    for _, row in sentiment_counts_by_status.iterrows():
        file.write(
            f"Status: {row['status']}, Sentiment: {row['sentiment_category']}, Proportion: {row['proportion']:.2%}\n"
        )
    file.write("\n")
    
    file.write("Relative distribution of sentiment in P.P. vs C.G:\n")
    for _, row in grouped_data.iterrows():
        file.write(
            f"Group: {row['group']}, Sentiment: {row['sentiment_category']}, Relative Frequency: {row['relative_freq']:.2%}\n"
        )
    file.write("\n")
    
# Calcolo overlap di parole tra i vari status

# Funnzione per estrarre le parole uniche per status
def get_words(texts):
    words = set()
    for text in texts:
        words.update(text.split())
    return words

words_by_status = {
    status: get_words(texts) for status,
    texts in texts_by_status.items()}

# Calcolo overlap tra le parole di ogni coppia di status
overlap_by_status_pair = {}

# Creazione coppie di status
statuses = list(statuses)
for i in range(len(statuses)):
    for j in range(i+1, len(statuses)):
        status1, status2 = statuses[i], statuses[j]
        overlap = words_by_status[status1].intersection(words_by_status[status2])
        overlap_by_status_pair[(status1, status2)] = overlap

# Emotion detection
# Dizionario di synset per emozioni

emotion_keywords = [
    "happiness", "sadness", "anger", "fear", "disgust", "surprise", "trust",
    "anticipation"
]

# Funzione per ottenere i synsets per ogni macro-emozione da WordNet
def get_emotion_synsets(emotion_keywords):
    emotion_synsets = {}
    for keyword in emotion_keywords:
        synsets = wn.synsets(keyword)  # Trova tutti i synsets di ogni emozione
        synset_names = [s.name() for s in synsets if s.pos() == 'n']  # Filtra solo i sostantivi (nouns)
        if synset_names:
            emotion_synsets[keyword] = synset_names
    return emotion_synsets

# Ottieni i synsets
emotion_synsets = get_emotion_synsets(emotion_keywords)

# Funzione per ottenere tutte le parole emozione dei synset di ciascuna macro-emozione
def get_words_from_emotion_synsets(dic):
    emotion_words = {}
    for emotion, synsets in dic.items():
        words = []
        for synset_name in synsets:
            try:
                synset = wn.synset(synset_name)
                words.extend(synset.lemma_names())
            except Exception as e:
                print(f"Errore con il synset {synset_name}: {e}")
        emotion_words[emotion] = set(words)
    return emotion_words

# Estrazione delle parole emozione dai synset
emotion_lexicon = get_words_from_emotion_synsets(emotion_synsets)

# Normalizzazione dei risultati
emotion_lex_norm = {emotion.lower(): {word.lower() for word in words}
                    for emotion, words in emotion_lexicon.items()}

# Funzione per calcolare la frequenza di parole emozione
def count_words_per_emotion(sentence, emotion_words):
    tokens = word_tokenize(sentence)
    word_counts = {emotion: Counter() for emotion in emotion_words} 
    for emotion, words in emotion_words.items():
        word_counts[emotion].update([token for token in tokens if token in words])
    return word_counts

# Dizionario che contiene i risultati
results = {}

# Calcolo frequenza parole emozione per enunciato
for status in statuses:
    texts = data[data['status'] == status]['processed_text']
    total_word_counts = Counter()

    for text in texts:
        word_counts = count_words_per_emotion(text, emotion_lex_norm)
        for emotion_counts in word_counts.values():
            total_word_counts.update(emotion_counts)

    results[status] = total_word_counts 
    
# Funzione per vettore di frequenze di parole emozione in base allo status
def create_vector_for_status(word_counts, all_words):
    vector = np.zeros(len(all_words))
    word_to_index = {word: idx for idx, word in enumerate(all_words)}
    for word, count in word_counts.items():
        if word in word_to_index:
            vector[word_to_index[word]] = count
    return vector

# Insieme di parole emozione e creazione vettore per ogni stato
all_words = set(word for word_counts in results.values()
                for word in word_counts.keys())

vectors = {
    status: create_vector_for_status(
        word_counts,
        all_words) for status,
    word_counts in results.items()}

# Calcolo matrice similarità coseno tra ogni coppia di status
similarity_matrix = cosine_similarity(list(vectors.values()))

similarities = []

for i, status1 in enumerate(results.keys()):
    for j, status2 in enumerate(results.keys()):
        if i != j:
            similarities.append((status1, status2, similarity_matrix[i][j]))

with open('similarities.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Status1', 'Status2', 'Cosine Similarity'])
    writer.writerows(similarities)

# Calcolo coppie di status con similarità massima e minima
max_similarity = max(similarities, key=lambda x: x[2])
min_similarity = min(similarities, key=lambda x: x[2])


# CREAZIONE GRAFICI

# Grafico di distribuzione relativa del sentiment per status
plt.figure(figsize=(9, 4))
sns.barplot(
    x='status',
    y='proportion',
    hue='sentiment_category',
    data=sentiment_counts_by_status)

plt.title('Relative distribution of sentiment by status', fontsize=16)
plt.xlabel('Status', fontsize=14)
plt.ylabel('Relative frequency', fontsize=14)
plt.legend(title='Sentiment category')

plt.savefig('sent_distr_plot.png', bbox_inches='tight')

plt.tight_layout()

plt.close()

# Gafico distribuzione relativa del sentiment "Popolazione di Pazienti" vs. "Gruppo di Controllo"

plt.figure(figsize=(9, 4))
sns.barplot(
    x='group',
    y='relative_freq',
    hue='sentiment_category',
    data=grouped_data)

plt.title('Relative distribution of sentiment in P.P. vs C.G', fontsize=16)
plt.xlabel('Groups', fontsize=14)
plt.ylabel('Relative frequency', fontsize=14)
plt.legend(title='Sentiment Category')

plt.savefig('sdp_PPvsCG.png', bbox_inches='tight')

plt.tight_layout()

plt.close()

# Diagrammi di Venn per overlap in ogni coppia di status
for (status1, status2), overlap in overlap_by_status_pair.items():
    plt.figure(figsize=(6, 6))
    venn2(subsets=(len(words_by_status[status1]), len(words_by_status[status2]), len(overlap)), 
          set_labels=(status1, status2))
    
    plt.title(f"Word overlap: {status1} vs {status2}")
    
    plt.savefig(f"venn_{status1}_{status2}.png", bbox_inches='tight')
    plt.close()
    
# Grafico di parole emozione più frequenti per ogni status
for status, word_counts in results.items():
    top_10 = word_counts.most_common(10)
    if top_8:
        words, frequencies = zip(*top_10)
        plt.figure(figsize=(10, 6))
        plt.bar(words, frequencies, color='purple')
        plt.xlabel("Emotion words")
        plt.ylabel("Frequency")
        plt.title(f"Most frequent emotion words for '{status}'")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"emotion_words_{status}.png", bbox_inches='tight')
        plt.close()

# Creazione di un network basato sulla similarità coseno tra i vettori degli stati
G = nx.Graph()

# Archi del network in base ai pesi delle similarità coseno
for node1, node2, similarity in similarities:
    G.add_edge(node1, node2, weight=similarity)

pos = nx.circular_layout(G)

plt.figure(figsize=(10, 8))

# Nodi e archi
nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=3000,
    node_color='pink',
    font_size=14,
    font_weight='bold',
    edge_color='gray',
    linewidths=2,
    font_color='black')

edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=edge_labels,
    font_size=12,
    font_color='black')

plt.gcf().subplots_adjust(right=0.9)
plt.gca().add_patch(
    plt.Rectangle(
        (0.76,
         0.5),
        0.23,
        0.2,
        color='none',
        alpha=0.5,
        lw=2,
        edgecolor='black'))

plt.text(
    0.77,
    0.75,
    f"Max Similarity: {max_similarity[2]:.4f}\nbetween {max_similarity[0]} and {max_similarity[1]}",
    ha='left',
    va='top',
    fontsize=12,
    color='black')
plt.text(
    0.77,
    0.65,
    f"Min Similarity: {min_similarity[2]:.4f}\nbetween {min_similarity[0]} and {min_similarity[1]}",
    ha='left',
    va='top',
    fontsize=12,
    color='black')

plt.title("Cosine similarity graph", fontsize=16, fontweight='bold')

plt.savefig('cos_sim_graph.png', bbox_inches='tight')

plt.close()

# TESTING DELLE FUNZIONI

class Test_Functions(unittest.TestCase):

    def test_preprocess_text(self):
        sample_text = "Hello! It is raining in Belgium."
        processed = preprocess_text(sample_text)
        tokens = word_tokenize(processed)

        self.assertNotIn("in", tokens)
        self.assertNotIn("!", processed)
        self.assertIn("hello", tokens)


    def test_get_most_freq_words(self):
        sample_texts = ["i feel sad i feel happy i am happy"]
        top_words = get_most_freq_words(sample_texts, n=3)

        self.assertTrue(isinstance(top_words, list))
        self.assertGreater(len(top_words), 0)
        self.assertEqual(top_words[0][0], "i")
        self.assertEqual(top_words[0][1], 3)
        self.assertEqual(top_words[1][0], "feel")
        self.assertEqual(top_words[1][1], 2)

    def test_get_most_freq_bigrams(self):
        preprocessed_texts = [
            "feel sad hopeless",
            "feel sad hopeless again",
            "cannot go making feel worse"
        ]

        expected_bigrams = [
            (('feel', 'sad'), 2),
            (('sad', 'hopeless'), 2),
            (('hopeless', 'again'), 1),
            (('cannot', 'go'), 1),
            (('go', 'making'), 1),
            (('making', 'feel'), 1),
            (('feel', 'worse'), 1)
        ]
        
        result = get_most_freq_bigrams(preprocessed_texts, n=10)
        self.assertEqual(result, expected_bigrams)
              
    def test_get_sentiment_score(self):
        sia = SentimentIntensityAnalyzer()
        sample_text = "I am feeling very happy today."
        sentiment_score = get_sentiment_score(sample_text)

        self.assertGreaterEqual(sentiment_score, -1)
        self.assertLessEqual(sentiment_score, 1)
        self.assertGreater(sentiment_score, 0)

    def test_get_emotion_synsets(self):
        emotion_keywords = ['happiness', 'sadness', 'anger']
        emotion_synsets = get_emotion_synsets(emotion_keywords)

        self.assertGreater(len(emotion_synsets), 0)

        for keyword in emotion_keywords:
            self.assertIn(keyword, emotion_synsets)
            
        for synsets in emotion_synsets.values():
            for synset_name in synsets:
                synset = wn.synset(synset_name)
                self.assertEqual(synset.pos(), 'n') 
        
    def test_get_words_from_emotion_synsets(self):
        emotion_keywords = ['happiness', 'sadness', 'anger']

        synsets_dic = get_emotion_synsets(emotion_keywords)

        actual_output = get_words_from_emotion_synsets(synsets_dic)
        self.assertTrue(actual_output)
        
        for emotion, words in actual_output.items():
            self.assertTrue(words)

        self.assertIn('felicity', actual_output['happiness'])
        self.assertIn('sorrow', actual_output['sadness'])
        self.assertIn('anger', actual_output['anger'])

    def test_get_words(self):
        sample_texts = [
            "this is a test",
            "blue is a nice colour",
            "this test is nice"]
        expected_words = {"this", "is", "a", "test", "blue", "colour", "nice"}
        result = get_words(sample_texts)

        self.assertEqual(result, expected_words)

    def test_count_words_per_emotion(self):
        emotion_words = {
            'happy': {
                'happy', 'joy', 'excited'}, 'sad': {
                'sad', 'unhappy', 'cry'}, 'angry': {
                'angry', 'mad', 'furious'}}
        sentence = "I feel very happy and excited, but also a bit sad and furious."
        expected_output = {'happy': Counter({'happy': 1, 'excited': 1}), 'sad': Counter({
            'sad': 1}), 'angry': Counter({'furious': 1})}

        self.assertEqual(
            count_words_per_emotion(
                sentence,
                emotion_words),
            expected_output)

    def test_create_vector_for_status(self):
        word_counts = Counter({'happy': 2, 'sad': 1})
        all_words = ['happy', 'sad', 'angry']
    
        vector = create_vector_for_status(word_counts, all_words)
    
        self.assertEqual(vector[0], 2)
        self.assertEqual(vector[1], 1)
        self.assertEqual(vector[2], 0)

if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)