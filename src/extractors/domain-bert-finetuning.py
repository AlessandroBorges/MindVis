from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer
from datasets import Dataset
import torch
from typing import List, Dict
import numpy as np

class DomainBERTFineTuner:
    def __init__(
        self,
        base_model: str = 'neuralmind/bert-base-portuguese-cased',
        max_length: int = 512
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForMaskedLM.from_pretrained(base_model)
        self.max_length = max_length

    def prepare_dataset(
        self,
        texts: List[str],
        validation_split: float = 0.1
    ) -> Dict[str, Dataset]:
        """Prepara o dataset para fine-tuning"""
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True
            )

        # Cria dataset do Hugging Face
        dataset = Dataset.from_dict({'text': texts})
        
        # Tokeniza os textos
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text']
        )

        # Divide em treino e validação
        split_dataset = tokenized_dataset.train_test_split(
            test_size=validation_split
        )

        return {
            'train': split_dataset['train'],
            'validation': split_dataset['test']
        }

    def train(
        self,
        datasets: Dict[str, Dataset],
        output_dir: str,
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
    ):
        """Realiza o fine-tuning do modelo"""
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            eval_steps=500,
            save_steps=1000,
            evaluation_strategy="steps",
            load_best_model_at_end=True,
            save_total_limit=2,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets['train'],
            eval_dataset=datasets['validation'],
        )

        # Realiza o treinamento
        trainer.train()
        
        # Salva o modelo e tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

# Exemplo de uso
if __name__ == "__main__":
    # Exemplo com textos médicos
    medical_texts = [
        """O paciente apresentou quadro de hipertensão arterial sistêmica,
        com níveis pressóricos elevados e sinais de comprometimento cardíaco.
        Foram solicitados exames complementares incluindo ecocardiograma.""",
        
        """Avaliação mostrou diabetes mellitus tipo 2 descompensada,
        com glicemia de jejum elevada e hemoglobina glicada acima dos
        valores de referência.""",
        # ... adicionar mais textos do domínio
    ]

    # Inicializa o fine-tuner
    finetuner = DomainBERTFineTuner()
    
    # Prepara os datasets
    datasets = finetuner.prepare_dataset(medical_texts)
    
    # Realiza o fine-tuning
    finetuner.train(
        datasets=datasets,
        output_dir='./medical_bert',
        num_epochs=3,
        batch_size=8
    )
