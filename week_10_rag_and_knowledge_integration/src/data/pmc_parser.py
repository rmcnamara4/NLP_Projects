import xml.etree.ElementTree as ET 
from typing import Dict, List, Optional

def _get_text(elem: Optional[ET.Element]) -> str: 
    """
    Extracts and normalizes the text content from an XML element.

    This function retrieves all text contained within the given XML element
    (including nested children), joins it into a single string, and normalizes
    whitespace by collapsing consecutive spaces and trimming leading/trailing
    whitespace.

    Args:
        elem (Optional[xml.etree.ElementTree.Element]): 
            The XML element from which to extract text. If None, returns an empty string.

    Returns:
        str: The normalized text content of the element, or an empty string if the element is None.
    """
    if elem is None: 
        return ''
    return ' '.join(''.join(elem.itertext()).split())

def _join_text(elems: List[ET.Element]) -> str: 
    """
    Joins and normalizes text content from a list of XML elements.

    For each element in the input list, extracts its text content using
    `_get_text`, skips None elements, and joins the resulting strings
    with two newlines between them. Leading and trailing whitespace is
    trimmed from the final result.

    Args:
        elems (List[xml.etree.ElementTree.Element]):
            A list of XML elements to extract and combine text from. 
            Elements that are None are ignored.

    Returns:
        str: The combined and normalized text content of all elements,
        separated by blank lines. Returns an empty string if no valid 
        elements are provided.
    """
    return '\n\n'.join([_get_text(e) for e in elems if e is not None]).strip()

def parse_pmc_xml(xml_str: str) -> Dict: 
    """
    Parses a PubMed Central (PMC) XML string and extracts key article metadata.

    This function processes the given PMC-formatted XML string and retrieves
    common fields such as the article title, abstract, body text, identifiers,
    author list, journal, and publication date.

    Extracted fields:
        - title (str): Article title.
        - abstract (str): Combined abstract text.
        - body (str): Full body text, joined from paragraph nodes.
        - pmcid (str): PubMed Central identifier (if available).
        - pmid (str): PubMed identifier (if available).
        - doi (str): Digital Object Identifier (if available).
        - authors (List[str]): List of authors in "Given Surname" format.
        - journal (str): Journal title.
        - pub_date (str): Publication date in "YYYY-MM-DD" format (best available).

    Args:
        xml_str (str):
            A string containing the PMC XML document.

    Returns:
        Dict: A dictionary containing extracted metadata and text content.
    """
    root = ET.fromstring(xml_str) 

    title = _get_text(root.find('.//front//article-title'))
    abstract = _join_text(root.findall('.//front//abstract'))

    body_paras = root.findall('.//body//p') 
    body_text = _join_text(body_paras) 

    pmcid = _get_text(root.find('.//article-id[@pub-id-type="pmcid"]'))
    pmid  = _get_text(root.find('.//article-id[@pub-id-type="pmid"]'))
    doi   = _get_text(root.find('.//article-id[@pub-id-type="doi"]'))

    authors = []
    for a in root.findall('.//contrib-group//contrib[@contrib-type="author"]'): 
        given = _get_text(a.find('.//given-names'))
        surname = _get_text(a.find('.//surname'))
        if given or surname: 
            authors.append(f'{given} {surname}'.strip())

    journal = _get_text(root.find('.//journal//journal-title'))

    pub_date_elem = root.find('.//pub-date[@pub-type="epub"]') or root.find('.//pub-date[@pub-type="ppub"]') or root.find('.//pub-date')
    pub_date = ''
    if pub_date_elem is not None: 
        y = _get_text(pub_date_elem.find('year'))
        m = _get_text(pub_date_elem.find('month')) 
        d = _get_text(pub_date_elem.find('day'))
        pub_date = '-'.join(x for x in [y, m, d] if x) 

    return {
        'title': title, 
        'abstract': abstract, 
        'body': body_text, 
        'pmcid': pmcid, 
        'pmid': pmid, 
        'doi': doi, 
        'authors': authors, 
        'journal': journal, 
        'pub_date': pub_date
    }

def parse_many(xml_strings: List[str]) -> List[Dict]: 
    """
    Parses multiple PubMed Central (PMC) XML strings and extracts metadata.

    Each XML string is processed with `parse_pmc_xml`. Invalid XML strings
    that raise `xml.etree.ElementTree.ParseError` are skipped.

    Args:
        xml_strings (List[str]):
            A list of PMC-formatted XML strings to parse.

    Returns:
        List[Dict]: A list of dictionaries containing extracted article
        metadata and text content. Entries corresponding to invalid XML
        inputs are omitted.
    """
    results = []
    for x in xml_strings: 
        try: 
            results.append(parse_pmc_xml(x))
        except ET.ParseError: 
            pass
    return results
    