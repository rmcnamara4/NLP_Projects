import re
import html

_WS = re.compile(r'\s+')

# Precompiled patterns (kept fairly conservative)
_BRACKET_CITES = re.compile(r'\[\d+(?:\s*,\s*\d+)*\]')  # [1] or [1, 2, 10]
_FIG_REFS = re.compile(r'\(Fig(?:ure)?\.?\s*\d+[a-z]?\)', re.I)  # (Fig. 3), (Figure 5a)
_YEAR_PARENS = re.compile(r'\(\s*\d{4}[a-z]?\s*\)')  # (2021), (2021a)

# (Smith, 2020), (Smith & Jones, 2021), (Smith et al., 2019; Jones, 2020)
_PAREN_AUTHOR_YEAR = re.compile(
    r'''
    \(
      [^()]*?                # minimal stuff (author list)
      [A-Z][^()]*?          # ensure it starts like a name (capitalized)
      ,\s*\d{4}[a-z]?       # , 2020 / , 2020a
      (?:[^()]*?;\s*[^()]*?\d{4}[a-z]?)*  # ; Jones, 2021 (optional repeats)
    \)
    ''',
    re.X,
)

# Smith (2020), Smith & Jones (2021), Smith et al. (2019)
_LEAD_AUTHOR_YEAR = re.compile(
    r'''
    \b
    (?:[A-Z][A-Za-z\-']+      # Smith
       (?:\s*&\s*[A-Z][A-Za-z\-']+)?    # & Jones (optional)
       (?:\s+et\ al\.)?       # et al. (optional)
    )
    \s*\(\s*\d{4}[a-z]?\s*\)  # (2020) / (2020a)
    ''',
    re.X,
)

_HTML_TAGS = re.compile(r'<[^>]+>')  # crude but effective tag stripper


def clean_text(
    text: str,
    *,
    remove_html: bool = True,
    remove_bracket_cites: bool = True,
    remove_figure_refs: bool = True,
    remove_year_parens: bool = True,
    remove_author_year: bool = True,
    unescape_html_entities: bool = True,
) -> str:
    if not text:
        return ''

    if unescape_html_entities:
        text = html.unescape(text)

    if remove_html:
        text = _HTML_TAGS.sub(' ', text)

    if remove_bracket_cites:
        text = _BRACKET_CITES.sub(' ', text)

    if remove_figure_refs:
        text = _FIG_REFS.sub(' ', text)

    if remove_year_parens:
        text = _YEAR_PARENS.sub(' ', text)

    if remove_author_year:
        # Order matters: remove parenthetical first, then “Smith (2020)”
        text = _PAREN_AUTHOR_YEAR.sub(' ', text)
        text = _LEAD_AUTHOR_YEAR.sub(' ', text)

    # Final tidy
    text = _WS.sub(' ', text).strip()
    
    return text