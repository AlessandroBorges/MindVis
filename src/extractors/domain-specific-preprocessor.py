import spacy
from spacy.tokens import Doc
from spacy.language import Language
import re
from typing import List, Set, Dict

class DomainSpecificPreprocessor:
    def __init__(self, domain: str = 'medical'):
        self.nlp = spacy.load('pt_core_news_lg')
        self.domain = domain
        
        # Dicionário de termos técnicos por domínio
        self.domain_terms = {
            'medical': {
                'compound_terms': {
                    'pressão arterial': 'pressão_arterial',
                    'diabetes mellitus': 'diabetes_mellitus',
                    'infarto agudo': 'infarto_agudo',
                },
                'abbreviations': {
                    'PA': 'pressão_arterial',
                    'DM': 'diabetes_mellitus',
                    'IAM': 'infarto_agudo_miocárdio',
                },
                'prefix_patterns': [
                    'anti[\\-\\s]?\\w+',  # matches anti-inflamatório, antiviral
                    'cardio[\\-\\s]?\\w+', # matches cardiovascular
                    'hiper[\\-\\s]?\\w+'  # matches hipertensão
                ]
            },
            'legal': {
                'compound_terms': {
                    'direito administrativo': 'direito_administrativo',
                    'servidor público': 'servidor_público',
                    'processo administrativo': 'processo_administrativo'
                },
                'abbreviations': {
                    'CF': 'constituição_federal',
                    'STF': 'supremo_tribunal_federal',
                    'MP': 'ministério_público'
                },
                'prefix_patterns': [
                    'sub[\\-\\s]?\\w+',  # matches sub-rogação
                    'contra[\\-\\s]?\\w+', # matches contraditório
                    'auto[\\-\\s]?\\w+'  # matches auto-executoriedade
                ]
            }
        }
        
        # Registra componente customizado no pipeline do spaCy
        if not Language.has_factory('domain_merger'):
            @Language.factory('domain_merger')
            def create_domain_merger(nlp, name):
                return DomainTermMerger(nlp, self.domain_terms[self.domain])
            
        # Adiciona o componente ao pipeline
        if 'domain_merger' not in self.nlp.pipe_names:
            self.nlp.add_pipe('domain_merger', after='ner')

    def preprocess(self, text: str) -> Doc:
        """Preprocessa o texto aplicando regras específicas do domínio"""
        # Normaliza abreviações
        text = self._normalize_abbreviations(text)
        
        # Aplica o pipeline do spaCy com as regras customizadas
        doc = self.nlp(text)
        
        return doc

    def _normalize_abbreviations(self, text: str) -> str:
        """Normaliza abreviações conhecidas do domínio"""
        normalized = text
        for abbr, full in self.domain_terms[self.domain]['abbreviations'].items():
            pattern = r'\b' + re.escape(abbr) + r'\b'
            normalized = re.sub(pattern, full, normalized)
        return normalized

class DomainTermMerger:
    def __init__(self, nlp, domain_terms):
        self.domain_terms = domain_terms
        
    def __call__(self, doc):
        """Merge tokens that form domain-specific terms"""
        with doc.retokenize() as retokenizer:
            for compound_term in self.domain_terms['compound_terms']:
                matches = re.finditer(r'\b' + re.escape(compound_term) + r'\b', doc.text.lower())
                for match in matches:
                    start_token = doc.char_span(match.start(), match.end())
                    if start_token is not None:
                        retokenizer.merge(start_token)
        return doc

# Exemplo de uso
if __name__ == "__main__":
    # Exemplo para domínio médico
    medical_preprocessor = DomainSpecificPreprocessor(domain='medical')
    medical_text = """
    O paciente apresentou PA elevada e DM tipo 2. 
    Foi prescrito anti-inflamatório e observada condição cardiovascular.
    """
    doc_medical = medical_preprocessor.preprocess(medical_text)
    print("Tokens médicos processados:", [token.text for token in doc_medical])
    
    # Exemplo para domínio jurídico
    legal_preprocessor = DomainSpecificPreprocessor(domain='legal')
    legal_text = """
    O STF decidiu sobre processo administrativo disciplinar.
    O MP apresentou contra-razões no caso de auto-executoriedade.
    """
    doc_legal = legal_preprocessor.preprocess(legal_text)
    print("Tokens jurídicos processados:", [token.text for token in doc_legal])
