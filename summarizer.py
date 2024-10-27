import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import networkx as nx

def create_graph_sentence(sentences):
    G = nx.Graph()

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            Jaccard_similarity = len(set(sentences[i]) & set(sentences[j])) / len(set(sentences[i]) | set(sentences[j]))
            G.add_edge(i, j, weight=Jaccard_similarity)
    return G
    
def generate_text_rank_summary(text, num_sentences=50):
    sentences = [word_tokenize(sentence) for sentence in sent_tokenize(text)]
    sentence_graph = create_graph_sentence(sentences)
    scores = nx.pagerank(sentence_graph)
    sort_sentences = sorted(scores, key=scores.get, reverse=True)[:num_sentences]
    sentences_chosen = [sentences[i] for i in sort_sentences]
    summary = " ".join(TreebankWordDetokenizer().detokenize(sentence) for sentence in sentences_chosen)
    return summary
