from typing import List, Tuple
from langchain_community.embeddings import OllamaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embedder = OllamaEmbeddings(model='nomic-embed-text') # ollama pull nomic-embed-text

def split_text_into_sentences(text: str) -> List[str]:
    raw_sentences = text.replace('\n', ' ').split(".")
    return [s.strip() for s in raw_sentences if s.strip()]

def embed(sentence: str):
    return embedder.embed_query(sentence)

def calculate_similarity(reference_sentence: str, sentences: List[str]) -> List[Tuple[float, str]]:
    ref_vector = embed(reference_sentence)
    similarities = []

    for sentence in sentences:
        sent_vector = embed(sentence)
        similarity_score = cosine_similarity([ref_vector], [sent_vector])[0][0]
        similarities.append((similarity_score, sentence))

    return similarities

def reorder_sentences_by_similarity(similarities: List[Tuple[float, str]]) -> List[Tuple[float, str]]:
    return sorted(similarities, key=lambda x: x[0], reverse=True)

if __name__ == '__main__':
    long_text = """
    O olhar da natureza é um livro aberto cheio de segredos e mistérios. A floresta silvestre é um lugar 
    onde o tempo parece parar, onde as árvores se erguem como colunas naturais, e os rios correm como veias vitais. 
    O som das aves é uma música celestial que nos leva a momentos de paz e reflexão.
    """

    reference_sentence = 'Arvores são colunas naturais'

    sentences = split_text_into_sentences(long_text)
    similarities = calculate_similarity(reference_sentence, sentences)
    ordered_sentences = reorder_sentences_by_similarity(similarities)

    print('Sentenças organizadas por similaridade:')
    for score, sentence in ordered_sentences:
        print(f'Similaridade: {score:.2f}% - {sentence}')