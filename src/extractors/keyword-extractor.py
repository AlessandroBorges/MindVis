import spacy
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from keybert import KeyBERT
import yake

class KeywordExtractor:
    def __init__(self, language='pt'):
        # Carrega o modelo do spaCy para português
        self.nlp = spacy.load('pt_core_news_lg')
        # Inicializa o KeyBERT (baseado em transformers)
        self.kw_model = KeyBERT()
        # Configuração do YAKE
        self.kw_extractor = yake.KeywordExtractor(
            lan=language,
            n=3,  # ngrams
            dedupLim=0.7,
            windowsSize=1,
            top=20
        )

    def preprocess_text(self, text):
        """Pré-processamento básico do texto"""
        doc = self.nlp(text)
        # Remove pontuação e stopwords, mantém apenas substantivos, adjetivos e verbos
        tokens = [token.lemma_.lower() for token in doc 
                 if not token.is_stop 
                 and not token.is_punct
                 and token.pos_ in ['NOUN', 'ADJ', 'VERB']]
        return ' '.join(tokens)

    def extract_keywords_tfidf(self, texts, top_n=10):
        """Extração usando TF-IDF"""
        vectorizer = TfidfVectorizer(
            preprocessor=self.preprocess_text,
            ngram_range=(1, 3)
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Obtém as palavras com maiores scores TF-IDF
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.sum(axis=0).A1
        keywords = [(feature_names[i], scores[i]) 
                   for i in scores.argsort()[-top_n:][::-1]]
        return keywords

    def extract_keywords_keybert(self, text, top_n=10):
        """Extração usando KeyBERT (baseado em BERT)"""
        keywords = self.kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words='portuguese',
            top_n=top_n
        )
        return keywords

    def extract_keywords_yake(self, text, top_n=10):
        """Extração usando YAKE"""
        keywords = self.kw_extractor.extract_keywords(text)
        return keywords[:top_n]

    def extract_keywords_ensemble(self, texts, top_n=10):
        """Combina diferentes métodos de extração"""
        all_keywords = []
        
        # Aplica cada método
        for text in texts:
            keywords_tfidf = self.extract_keywords_tfidf([text], top_n)
            keywords_keybert = self.extract_keywords_keybert(text, top_n)
            keywords_yake = self.extract_keywords_yake(text, top_n)
            
            # Combina os resultados
            all_keywords.extend([kw[0] for kw in keywords_tfidf])
            all_keywords.extend([kw[0] for kw in keywords_keybert])
            all_keywords.extend([kw[0] for kw in keywords_yake])
        
        # Conta frequência das palavras-chave
        keyword_freq = Counter(all_keywords)
        
        # Retorna as palavras-chave mais frequentes
        return keyword_freq.most_common(top_n)

# Exemplo de uso
if __name__ == "__main__":
    texts = [
        """O Direito Administrativo é o ramo do direito público que estuda
        os princípios e regras que disciplinam a função administrativa e
        que abrange entes, órgãos, agentes e atividades públicas.""",
        
        """A Administração Pública deve seguir os princípios constitucionais
        da legalidade, impessoalidade, moralidade, publicidade e eficiência."""
    ]
    
    extractor = KeywordExtractor()
    keywords = extractor.extract_keywords_ensemble(texts)
    print("Palavras-chave encontradas:", keywords)
