from jinja2 import Environment, FileSystemLoader, select_autoescape
from functools import lru_cache
from typing import List

@lru_cache(maxsize = 1) 
def _env(dir: str) -> Environment: 
    """
    Create and cache a Jinja2 environment with consistent configuration. 

    The environment is cached (LRU, size=1) so repeated calls return the same 
    environment instance for the given template directory. Configuration includes 
    file-system loading, selective autoescaping for `.j2` files, and clean block handling.

    Args:
        dir (str): Directory containing Jinja2 template files.

    Returns:
        Environment: Configured Jinja2 environment instance.
    """
    return Environment(
        loader = FileSystemLoader(dir), 
        autoescape = select_autoescape(enabled_extensions = ('j2', ), default_for_string = False), 
        trim_blocks = True, 
        lstrip_blocks = True
    )

def render_template(dir: str, name: str, **vars) -> str: 
    """
    Load and render a Jinja2 template with the provided variables.  

    Args:
        dir (str): Directory containing Jinja2 template files.  
        name (str): Name of the template file to render.  
        **vars: Arbitrary keyword arguments passed as variables to the template.  

    Returns:
        str: Rendered template as a string.  
    """
    template = _env(dir).get_template(name) 
    return template.render(**vars) 

def build_prompt(dir: str, question: str, contexts: List[str]) -> str: 
    """
    Construct a RAG-style prompt by rendering the 'rag_answer.j2' template.  

    Args:
        dir (str): Directory containing the Jinja2 templates.  
        question (str): Input question to be answered.  
        contexts (List[str]): Retrieved context passages to include in the prompt.  

    Returns:
        str: Rendered prompt string including the question and supporting contexts.  
    """
    return render_template(
        dir, 
        'rag_answer.j2', 
        question = question, 
        contexts = contexts,
        citation_example = '1'
    )

def build_system_prompt(dir: str) -> str: 
    """
    Render the system prompt template for medical domain tasks.  

    Args:
        dir (str): Directory containing the Jinja2 templates.  

    Returns:
        str: Rendered system prompt string from the 'system_medical.j2' template.  
    """
    return render_template(dir, 'system_medical.j2') 