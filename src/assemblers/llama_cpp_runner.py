from llama_cpp import Llama
from typing import List, Dict, Tuple
import json

# Template do Llama 2 Chat
B_INST, E_INST = "<|start_header_id|>user<|end_header_id|>\n\n", "<|eot_id|>"
B_SYS, E_SYS = "<|start_header_id|>system<|end_header_id|>\n\n", "<|eot_id|>"
B_ASSIST, E_ASSIST = "<|start_header_id|>assistant<|end_header_id|>\n\n", "<|eot_id|>" 
BOS = '<|begin_of_text|>'
EOS = "<|end_of_text|>"

# Sistema prompt padrão do Llama 2
DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Be concise and reply in Brazilian Portuguese."


class LlamaLogprobExtractor:
    def __init__(
        self, 
        model_path: str,
        n_ctx: int = 1024,
        n_gpu_layers: int = 0
    ):
        """
        Inicializa o extrator de logprobs
        
        Args:
            model_path: Caminho para o modelo .gguf
            n_ctx: Tamanho do contexto
            n_gpu_layers: Número de camadas para rodar na GPU (0 = CPU apenas)
        """
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            logits_all=True,  # Importante: habilita o cálculo de logits para todos os tokens
            stop=[BOS,EOS,B_INST, E_INST, "</s>","\n"],  # Stopwords importantes
            chat_format="llama-3"
        )
    
    def create_prompt(self, instruction: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        """Cria o prompt formatado para o Llama 2 Chat"""
        return f"{B_INST} {B_SYS}{system_prompt}{E_SYS}{instruction} {E_INST}"
        
    def get_logprobs_for_completion(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.2,
        top_k: int = 40,
        top_p: float = 0.95,
        min_p: float = 0.08,
        typical_p: float = 1.0,
        n_probs: int = 5  # Número de top tokens para retornar logprobs
    ) -> Tuple[str, List[Dict]]:
        """
        Gera uma completion e retorna os logprobs dos tokens mais prováveis.
        
        Returns:
            Tuple contendo:
            - Texto gerado
            - Lista de dicts com logprobs para cada posição
        """
        #prompt = self.create_prompt(prompt)
        # Gera a completion com todos os logits
        completion = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            typical_p=typical_p,
            logprobs=n_probs,  # Número de logprobs alternativos para cada token
            #echo=True  # Inclui o prompt na saída
        )
        
        # Extrai o texto gerado
        generated_text = completion['choices'][0]['text']
        print("\n\nTexto gerado: ",generated_text)
        # Processa os logprobs
        logprobs_data = []
        
        if 'logprobs' in completion['choices'][0]:
            logprobs = completion['choices'][0]['logprobs']
            
            for i, token_logprobs in enumerate(logprobs['token_logprobs']):
                if token_logprobs is None:
                    continue
                    
                entry = {
                    'token': logprobs['tokens'][i],
                    'token_logprob': token_logprobs,
                    'top_logprobs': logprobs['top_logprobs'][i]
                }
                logprobs_data.append(entry)
        
        return generated_text, logprobs_data
    
    def analyze_token_probabilities(
    self,
    logprobs_data: List[Dict],
    threshold: float = -2.0
    ) -> Dict:
        """
        Analisa os logprobs para identificar tokens de alta/baixa confiança
        """
        analysis = {
            'high_confidence_tokens': [],
            'low_confidence_tokens': [],
            'average_logprob': 0,
            'token_count': len(logprobs_data)
        }
        
        total_logprob = 0
        
        for entry in logprobs_data:
            logprob = float(entry['token_logprob'])  # Convert to native Python float
            token = entry['token']
            
            if logprob > threshold:
                analysis['high_confidence_tokens'].append((token, float(logprob)))  # Convert when adding to list
            else:
                analysis['low_confidence_tokens'].append((token, float(logprob)))  # Convert when adding to list
                
            total_logprob += logprob
            
        analysis['average_logprob'] = float(total_logprob / len(logprobs_data))  # Convert final average
        
        return analysis

# Exemplo de uso
if __name__ == "__main__":
    # Inicializa o extrator
    extractor = LlamaLogprobExtractor(
        model_path= "G:/cache/lmstudio/models/lmstudio-community/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q6_K.gguf",
        n_gpu_layers=0  # Ajuste conforme sua GPU
    )
    
    # Exemplo de prompt
    prompt = "Responda de forma suscinta e objetiva, completando a frase: A capital do Equador é "
    
    # Gera completion com logprobs
    generated_text, logprobs = extractor.get_logprobs_for_completion(
        prompt,       
        max_tokens=3,
        temperature=0.0
    )
    
    # Analisa os resultados
    analysis = extractor.analyze_token_probabilities(logprobs)
    
    # Imprime resultados   
    print("\n\nAnálise de probabilidades:")
    print("#### Analysis:\n",analysis)
    
    print(f"\nTokens com alta confiança: {len(analysis['high_confidence_tokens'])}")
    print(f"\nTokens com baixa confiança: {len(analysis['low_confidence_tokens'])}")
    print(f"\nTotal de tokens: {analysis['token_count']}")
    print(f"\nLogprobs médio: {analysis['average_logprob']:.3f}")
    print(json.dumps(analysis, indent=2, ensure_ascii=False))
    
    # Exemplo de tokens com alta confiança
    print("\nTokens com alta confiança:")
    for token, prob in analysis['high_confidence_tokens']:
        print(f"Token: {token}, LogProb: {prob:.3f}")