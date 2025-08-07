from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from src.data.preprocessing import preprocess

from omegaconf import OmegaConf

from peft import PeftModel, PeftConfig

import torch 
import os 

class SummarizerPipeline: 
    def __init__(self): 
        self.model_name = 'rmcnamara4/pegasus-middle-chunking-v1'

        peft_config = PeftConfig.from_pretrained(self.model_name) 
        base_model_name = peft_config.base_model_name_or_path

        self.tokenizer = PegasusTokenizer.from_pretrained(self.model_name, trust_remote_code = True)
        base_model = PegasusForConditionalGeneration.from_pretrained(base_model_name, trust_remote_code = True, use_safetensors = True) 
        self.model =  PeftModel.from_pretrained(base_model, self.model_name)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.model.to(self.device) 
        self.model.eval()

        cfg = OmegaConf.load('outputs/pegasus_middle_chunking_v1/.hydra/config.yaml') 
        self.chunk_generation_cfg = OmegaConf.to_container(cfg._generation_dict.chunk_generation, resolve = True) 
        self.final_generation_cfg = OmegaConf.to_container(cfg._generation_dict.final_generation) 
        self.data_cfg = OmegaConf.to_container(cfg.datamodule, resolve = True) 
    
    def summarize(self, article: str) -> str: 
        dummy_batch = {
            'article': [article], 
            'abstract': ['']
        }
        dummy_idx = [0]

        batch_output = preprocess(
            batch = dummy_batch, 
            idx = dummy_idx, 
            tokenizer = self.tokenizer, 
            chunk_len = self.data_cfg['chunk_len'], 
            stride = self.data_cfg['stride'], 
            min_len = self.data_cfg['min_len'], 
            max_len = self.data_cfg['max_len'], 
            num_keep = self.data_cfg['num_keep'], 
            train = False, 
            chunking_strategy = self.data_cfg['chunking_strategy'], 
            embedding_model = self.data_cfg['embedding_model_name']
        )

        max_len = max(len(x) for x in batch_output['input_ids']) if len(batch_output['input_ids'] > 0) else 0 
        if max_len == 0: 
            return 'Not long enough to summarize.'
        input_ids = [x + [self.tokenizer.pad_token_id] * (max_len - len(x)) for x in batch_output['input_ids']]
        attention_mask = [x + [0] * (max_len - len(x)) for x in batch_output['attention_mask']]

        input_ids = torch.tensor(input_ids).to(self.device)
        attention_mask = torch.tensor(attention_mask).to(self.device)

        with torch.no_grad(): 
            generated_batch_ids = self.model.generate(
                input_ids = input_ids, 
                attention_mask = attention_mask, 
                max_new_tokens = self.chunk_generation_cfg['max_new_tokens'], 
                num_beams = self.chunk_generation_cfg['num_beams'], 
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id, 
                early_stopping = self.chunk_generation_cfg.get('early_stopping', False), 
                do_sample = self.chunk_generation_cfg['do_sample'], 
                top_p = self.chunk_generation_cfg.get('top_p', None), 
                top_k = self.chunk_generation_cfg.get('top_k', None), 
                temperature = self.chunk_generation_cfg.get('temperature', None), 
                repetition_penalty = self.chunk_generation_cfg['repetition_penalty'], 
                length_penalty = self.chunk_generation_cfg['length_penalty'], 
                no_repeat_ngram_size = self.chunk_generation_cfg['no_repeat_ngram_size']
            )

        chunk_summaries = self.tokenizer.batch_decode(generated_batch_ids, skip_special_tokens = True) 
        
        combined_text = ' '.join(chunk_summaries) 
        combined_ids = self.tokenizer.encode(combined_text, return_tensors = 'pt', add_special_tokens = False).squeeze()

        if len(combined_ids) > self.tokenizer.model_max_length - self.final_generation_cfg['max_new_tokens'] - 25: 
            combined_ids = combined_ids[:self.tokenizer.model_max_length - self.final_generation_cfg['max_new_tokens'] - 25]

        input_dict = {
            'input_ids': combined_ids.unsqueeze(0).to(self.device), 
            'attention_mask': torch.ones_like(combined_ids).unsqueeze(0).to(self.device)
        }

        with torch.no_grad(): 
            final_output = self.model.generate(
                **input_dict, 
                max_new_tokens = self.chunk_generation_cfg['max_new_tokens'], 
                num_beams = self.chunk_generation_cfg['num_beams'], 
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id, 
                early_stopping = self.chunk_generation_cfg.get('early_stopping', False), 
                do_sample = self.chunk_generation_cfg['do_sample'], 
                top_p = self.chunk_generation_cfg.get('top_p', None), 
                top_k = self.chunk_generation_cfg.get('top_k', None), 
                temperature = self.chunk_generation_cfg.get('temperature', None), 
                repetition_penalty = self.chunk_generation_cfg['repetition_penalty'], 
                length_penalty = self.chunk_generation_cfg['length_penalty'], 
                no_repeat_ngram_size = self.chunk_generation_cfg['no_repeat_ngram_size']
            )

        final_summary = self.tokenizer.decode(final_output[0], skip_special_tokens = True)
        
        return final_summary


