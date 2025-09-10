import os, json, boto3 
from typing import List, Optional, Dict, Any 
from abc import ABC, abstractmethod

from dotenv import load_dotenv
load_dotenv()

Message = Dict[str, Any]

def bedrock_client(): 
    return boto3.client(
        'bedrock-runtime', region_name = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    )

# Define variables for identifying which family of model the chosen LLM belongs to 
FAM_ANTHROPIC = 'anthropic.claude'
FAM_COHERE = 'cohere.command'
FAM_META_LLAMA = 'meta.llama'
FAM_AMAZON_TITAN = 'amazon.titan'

def _get_family(model_id: str) -> str: 
    """
    Identify the LLM family based on its model ID.

    This function inspects the model identifier string and maps it to a 
    known provider family (Anthropic, Cohere, Meta LLaMA, or Amazon Titan). 
    If the model ID does not match any of these, it returns "other".

    Args:
        model_id (str): The model identifier (e.g., 
            "us.anthropic.claude-3-5-haiku-20241022-v1:0").

    Returns:
        str: The family constant string corresponding to the model 
            (e.g., "anthropic.claude", "cohere.command"), or "other".
    """
    mid = model_id.lower()
    if 'anthropic' in mid or 'claude' in mid: return FAM_ANTHROPIC
    if 'cohere' in mid: return FAM_COHERE
    if 'llama' in mid: return FAM_META_LLAMA
    if 'titan' in mid: return FAM_AMAZON_TITAN
    return 'other'

class LLM(ABC): 
    """
    Abstract base class for Large Language Model (LLM) clients.

    Defines a standard interface for generating text completions
    from a sequence of system/user messages. Subclasses should
    implement the `generate` method for specific providers.
    """
    @abstractmethod
    def generate(self, messages: List[Message], **overrides) -> Dict[str, Any]: 
        """
        Generate a model response given a list of messages.

        Args:
            messages (List[Message]): A sequence of messages, where each
                message is a dict with "role" ("system", "user", or
                "assistant") and "content" (str).
            **overrides: Optional provider-specific parameters such as
                temperature, max_tokens, etc.

        Returns:
            Dict[str, Any]: A response dictionary containing:
                - "text" (str): The model-generated text.
                - "usage" (dict, optional): Token usage metadata.
                - "raw" (dict, optional): Full raw provider response.
        """
        pass 

    def simple(self, user_text: str, system_text: Optional[str] = None, **kw) -> str: 
        """
        Generate a response from a simple user/system text prompt.

        Wraps `generate` by constructing messages from raw strings,
        making it easier to call for quick single-turn prompts.

        Args:
            user_text (str): The user input prompt text.
            system_text (Optional[str]): An optional system instruction
                (e.g., role or task definition).
            **kw: Optional overrides passed to `generate`.

        Returns:
            str: The model-generated text (extracted from the response).
        """
        msgs: List[Message] = []
        if system_text: 
            msgs.append({'role': 'system', 'content': system_text})
        msgs.append({'role': 'user', 'content': user_text})
        out = self.generate(msgs, **kw)
        return out.get('text', '')

class BedrockLLM(LLM): 
    """
    LLM client for AWS Bedrock-hosted models.

    Provides a unified interface for Anthropic (Claude), Cohere,
    Meta (LLaMA), and Amazon Titan families. Handles message
    formatting, request construction, and response parsing.
    """
    def __init__(
        self, 
        model_id: Optional[str] = None, 
        client = None,
        default: Optional[Dict[str, Any]] = None
    ): 
        self.client = client or bedrock_client()
        self.model_id = model_id or os.getenv('BEDROCK_MODEL_ID', 'us.anthropic.claude-3-5-haiku-20241022-v1:0')
        self.family = _get_family(self.model_id) 
        self.default = {
            'max_tokens': 700, 
            'temperature': 0.2, 
            'top_p': 0.95, 
            **(default or {})
        }

    def generate(self, messages = List[Message], **overrides) -> Dict[str, Any]: 
        """
        Generate text from a sequence of messages using an AWS Bedrock model.

        Args:
            messages (List[Message]): A list of messages in dict form with
                "role" ("system" or "user") and "content" (str).
            **overrides: Optional parameters such as `max_tokens`,
                `temperature`, or `top_p` that override defaults.

        Returns:
            Dict[str, Any]: A dictionary with:
                - "text" (str): The generated text output.
                - "usage" (dict, optional): Token usage metadata if provided.
                - "raw" (dict): Full parsed provider response for debugging.
        """
        params = {**self.default, **overrides}
        if self.family == FAM_ANTHROPIC: 
            body = {
                'anthropic_version': 'bedrock-2023-05-31', 
                'max_tokens': params['max_tokens'], 
                'temperature': params['temperature'], 
                'top_p': params['top_p'], 
                'messages': [
                    {
                        'role': m['role'], 
                        'content': [{'type': 'text', 'text': m['content']}]
                    }
                    for m in messages if m['role'] != 'system'
                ]
            }

            sys_msgs = [m['content'] for m in messages if m['role'] == 'system']
            if sys_msgs: 
                body['system'] = '\n\n'.join(sys_msgs)

            resp = self.client.invoke_model(
                modelId = self.model_id, 
                body = json.dumps(body).encode('utf-8'), 
                accept = 'application/json', 
                contentType = 'application/json'
            )
            data = json.loads(resp['body'].read())

            text = ''
            for blk in data.get('content', []) or []: 
                if blk.get('type') == 'text': 
                    text += blk.get('text', '') 
            return {'text': text, 'usage': data.get('usage', {}), 'raw': data}
        
        elif self.family == FAM_COHERE: 
            sys = '\n\n'.join([m['content'] for m in messages if m['role'] == 'system'])
            user = '\n\n'.join([m['content'] for m in messages if m['role'] == 'user'])

            body = {
                'max_tokens': params['max_tokens'], 
                'temperature': params['temperature'], 
                'p': params['top_p'], 
                'message': f'System: {sys}\n\nUser: {user}' if sys else user
            }

        elif self.family == FAM_META_LLAMA: 
            parts = ['<|begin_of_text|>']

            # System message (optional)
            system_msgs = [m['content'] for m in messages if m['role'] == 'system']
            if system_msgs:
                sys_text = '\n'.join(system_msgs)
                parts.append(
                    f'<|start_header_id|>system<|end_header_id|>\n{sys_text}<|eot_id|>'
                )

            # User messages (can be multiple)
            for m in messages:
                if m['role'] == 'user':
                    parts.append(
                        f'<|start_header_id|>user<|end_header_id|>\n{m["content"]}<|eot_id|>'
                    )

            # Assistant slot (where generation starts)
            parts.append('<|start_header_id|>assistant<|end_header_id|>')

            prompt = '\n'.join(parts)

            body = {
                'prompt': prompt, 
                'max_gen_len': params['max_tokens'],
                'temperature': params['temperature'], 
                'top_p': params['top_p']
            }

        elif self.family == FAM_AMAZON_TITAN:
            sys = '\n\n'.join([m['content'] for m in messages if m['role'] == 'system'])
            user = '\n\n'.join([m['content'] for m in messages if m['role'] == 'user'])
            body = {
                'inputText': (f'[SYSTEM]\n{sys}\n\n[USER]\n{user}' if sys else user), 
                'textGenerationConfig': {
                    'maxTokenCount': params['max_tokens'], 
                    'temperature': params['temperature'],
                    'topP': params['top_p']
                },
            }
        else: 
            joined = '\n\n'.join(f'{m["role"].upper()}: {m["content"]}' for m in messages)
            body = {'prompt': joined, 'max_tokens': params['max_tokens'], 'temperature': params['temperature']}

        response = self.client.invoke_model(
            modelId = self.model_id, 
            accept = 'application/json', 
            contentType = 'application/json', 
            body = json.dumps(body).encode('utf-8') 
        )

        data = json.loads(response['body'].read())
        text = (
            data.get('outputText')
            or data.get('generation') 
            or data.get('generated_text') 
            or data.get('results', [{}])[0].get('output_text', '') 
            or data.get('completions', [{}])[0].get('data', {}).get('text', '')
            or ''
        )

        return {'text': text, 'usage': data.get('usage', {}), 'raw': data}





        
