from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch 

def load_quantized_model(cfg): 
    if cfg.load_8bit and cfg.load_4bit: 
        raise ValueError("You can't load both 8-bit and 4-bit. Choose one.")
    
    if cfg.load_4bit: 
        bnb_config = BitsAndBytesConfig(
            load_in_4bit = True, 
            bnb_4bit_compute_dtype = torch.float16, 
            bnb_4bit_use_double_quant = True, 
            bnb_4bit_quant_type = 'nf4'
        )
    elif cfg.load_8bit: 
        bnb_config = BitsAndBytesConfig(
            load_in_8bit = True
        )
    else: 
        bnb_config = None

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name) 

    kwargs = {
        'device_map': cfg.device_map,
        'torch_dtype': torch.float16,
    }
    
    if bnb_config is not None:
        kwargs['quantization_config'] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kwargs)

    pipe = pipeline(
        'text-generation', 
        model = model, 
        tokenizer = tokenizer, 
        return_full_text = cfg.return_full_text, 
        device_map = cfg.device_map
    )

    return pipe

