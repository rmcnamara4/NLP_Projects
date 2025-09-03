from jinja2 import Environment, FileSystemLoader, select_autoescape
from functools import lru_cache
from typing import List

@lru_cache(maxsize = 1) 
def _env(dir: str) -> Environment: 
    return Environment(
        loader = FileSystemLoader(dir), 
        autoescape = select_autoescape(enabled_extensions = ('j2', ), default_for_string = False), 
        trim_blocks = True, 
        lstrip_blocks = True
    )

def render_template(dir: str, name: str, **vars) -> str: 
    template = _env(dir).get_template(name) 
    return template.render(**vars) 

def build_prompt(dir: str, question: str, contexts: List[str]) -> str: 
    return render_template(
        dir, 
        'rag_answer.j2', 
        question = question, 
        contexts = contexts,
        citation_example = '1'
    )

def build_system_prompt(dir: str) -> str: 
    return render_template(dir, 'system_medical.j2') 