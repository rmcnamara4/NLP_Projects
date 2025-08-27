import os, json, boto3 
from typing import List, Optional, Dict, Any 
from abc import ABC, abstractmethod

from dotenv import load_dotenv
load_dotenv()

def bedrock_client(): 
    return boto3.client(
        'bedrock-runtime', region_name = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    )

FAM_ANTHROPIC = 'anthropic.claude'
FAM_COHERE = 'cohere.command'
FAM_META_LLAMA = 'meta.llama'
FAM_AMAZON_TITAN = 'amazon.titan'

def _get_family(model_id: str) -> str: 
    mid = model_id.lower()
    if 'anthropic' in mid or 'claude' in mid: return FAM_ANTHROPIC
    if 'cohere' in mid: return FAM_COHERE
    if 'llama' in mid: return FAM_META_LLAMA
    if 'titan' in mid: return FAM_AMAZON_TITAN
    return 'other'

Message = Dict[str, Any]

class LLM(ABC): 
    @abstractmethod
    def generate(self, messages = List[Message], **overrides) -> Dict[str, Any]: 
        pass 

    def simple(self, user_text: str, system_text: Optional[str] = None, **kw) -> str: 
        msgs: List[Message] = []
        if system_text: 
            msgs.append({'role': 'system', 'content': system_text})
        msgs.append({'role': 'user', 'content': user_text})
        out = self.generate(msgs, **kw)
        return out.get('text', '')

class BedrockLLM(LLM): 
    def __init__(
        self, 
        model_id: Optional[str], 
        client = None
    ): 
        self.client = client or bedrock_client()
        self.model_id = model_id or os.getenv('BEDROCK_MODEL_ID', 'us.anthropic.claude-3-5-haiku-20241022-v1:0')
        self.family = _get_family(model_id) 
        self.default = {
            'max_tokens': 700, 
            'temperature': 0.2, 
            'top_p': 0.95, 
            **(default or {})
        }

    def generate(self, messages = List[Message], **overrides) -> Dict[str, Any]: 
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

            resp = client.invoke_model(
                modelId = model_id, 
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

            full_prompt = '\n'.join(parts)

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
            contentType = 'appplication/json', 
            body = json.dumps(body).encode('utf-8') 
        )

        data = json.loads(response['body'].read())
        text = {
            data.get('outputText')
            or data.get('generation') 
            or data.get('generated_text') 
            or data.get('results', [{}])[0].get('output_text', '') 
            or data.get('completions', [{}])[0].get('data', {}).get('text', '')
        }

        return {'text': text, 'usage': data.get('usage', {}), 'raw': data}





        
