def create_prompts(data, template): 
    """
    Renders a list of prompts using a Jinja2 template and a list of dictionaries.

    Args:
        data (List[Dict]): A list of dictionaries, where each dict contains variables for template rendering.
        template (Template): A Jinja2 Template object.

    Returns:
        List[str]: Rendered prompt strings.
    """
    return [template.render(**entry) for entry in data]


