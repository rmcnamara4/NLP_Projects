from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch 

def load_quantized_model(cfg): 
    """
    Loads a quantized language model pipeline for text generation using Hugging Face Transformers.

    Supports loading models in 4-bit or 8-bit precision via `BitsAndBytesConfig`, and returns a 
    `transformers.pipeline` for generation.

    Args:
        cfg (object): Configuration object containing the following attributes:
            - model_name (str): Name or path of the pretrained model to load.
            - load_4bit (bool): Whether to load the model in 4-bit quantized mode.
            - load_8bit (bool): Whether to load the model in 8-bit quantized mode.
            - device_map (str or dict): Device mapping strategy (e.g., "auto", or device placement dict).
            - return_full_text (bool): Whether the generation pipeline should return full text or just the new tokens.

    Returns:
        transformers.Pipeline: A Hugging Face text-generation pipeline with the quantized model loaded.

    Raises:
        ValueError: If both `load_4bit` and `load_8bit` are set to True.
    """
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

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code = True) 

    kwargs = {
        'device_map': cfg.device_map,
        'torch_dtype': torch.float16,
    }
    
    if bnb_config is not None:
        kwargs['quantization_config'] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, trust_remote_code = True, use_safetensors = True, **kwargs)

    pipe = pipeline(
        'text-generation', 
        model = model, 
        tokenizer = tokenizer, 
        return_full_text = cfg.return_full_text, 
        device_map = cfg.device_map
    )

    return pipe

