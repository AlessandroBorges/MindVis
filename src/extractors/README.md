# Extratores 

Conjunto de ferramentas para extração de palavras-chave de textos em domínio específico

A classe KeywordExtractor é um extrator de palavras-chave que combina várias técnicas modernas de NLP. 
Aqui estão os principais componentes e métodos utilizados:

1. **Pré-processamento**:
   - Usa spaCy para análise morfológica
   - Remove stopwords e pontuação
   - Mantém apenas substantivos, adjetivos e verbos relevantes
   - Aplica lematização para normalizar as palavras

2. **Métodos de extração**:
   - **TF-IDF**: Identifica termos importantes baseado em sua frequência no documento e no corpus
   - **KeyBERT**: Usa modelos BERT para extrair palavras-chave semanticamente relevantes
   - **YAKE**: Algoritmo não-supervisionado que considera características estatísticas do texto
   - **Ensemble**: Combina os resultados dos três métodos anteriores

3. **Características especiais**:
   - Suporte a n-gramas (1 a 3 palavras)
   - Configurado para português
   - Ranking de palavras-chave por relevância
   - Possibilidade de ajustar o número de palavras-chave retornadas



## Domínos específicos

Para usar em um domínio específico como medicina ou direito administrativo, será necessário:

1. Ajustar os parâmetros do preprocessamento para incluir termos técnicos específicos
2. Criar uma lista de stopwords específica do domínio
3. Treinar ou fine-tunar o modelo BERT em textos do domínio específico
4. Ajustar os pesos do ensemble para favorecer o método que funciona melhor para cada caso

**Benefícios esperados:**

1. **Melhor Reconhecimento de Termos Técnicos**:
   - Mantém a integridade de termos compostos importantes
   - Reconhece variações de termos técnicos
   - Lida apropriadamente com abreviações do domínio

2. **Melhor Compreensão Contextual**:
   - O modelo fine-tunado aprende as nuances específicas do domínio
   - Melhora a identificação de relações entre termos técnicos
   - Captura melhor a semântica específica do domínio

3. **Resultados mais Precisos**:
   - Reduz falsos positivos de termos genéricos
   - Melhora a identificação de palavras-chave realmente relevantes
   - Mantém a consistência terminológica

### Implementação

A atual implementação compreende as seguintes classes:
	- DomainSpecificPreprocessor - Pre-processador de termos: vide item 1 abaixo
	- DomainBERTFineTuner - Fine Tune para modelos BERT: vide item 3 abaixo
	

1. **Ajuste de Parâmetros de Pré-processamento para Termos Técnicos**

- **Termos Compostos**: O código identifica e mantém unidos termos técnicos compostos (ex: "pressão arterial" → "pressão_arterial")
- **Abreviações**: Normaliza abreviações comuns do domínio (ex: "PA" → "pressão_arterial")
- **Padrões de Prefixos**: Identifica termos técnicos baseados em prefixos comuns (ex: "anti-", "cardio-")
- **Pipeline Customizado**: Adiciona componentes específicos ao pipeline do spaCy para tratamento especial de termos do domínio


2. **Criar uma lista de stopwords específica do domínio**
TBD - a estudar

3. **Fine-tuning do BERT para Domínio Específico**

**3. Fine-tuning do BERT - Benefícios esperados:**

1. **Melhor Reconhecimento de Termos Técnicos**:
   - Mantém a integridade de termos compostos importantes
   - Reconhece variações de termos técnicos
   - Lida apropriadamente com abreviações do domínio

2. **Melhor Compreensão Contextual**:
   - O modelo fine-tunado aprende as nuances específicas do domínio
   - Melhora a identificação de relações entre termos técnicos
   - Captura melhor a semântica específica do domínio

3. **Resultados mais Precisos**:
   - Reduz falsos positivos de termos genéricos
   - Melhora a identificação de palavras-chave realmente relevantes
   - Mantém a consistência terminológica

- **Preparação dos Dados**: 
  - Tokenização específica para o domínio
  - Tratamento de textos longos
  - Divisão apropriada entre treino e validação

- **Processo de Treinamento**:
  - Usa o modelo base em português (neuralmind/bert-base-portuguese-cased)
  - Configurações de treinamento otimizadas para domínios específicos
  - Salvamento de checkpoints e avaliação durante o treinamento


4. **Ajustar os pesos do ensemble para Domínios específicos**
TBD - a estudar

