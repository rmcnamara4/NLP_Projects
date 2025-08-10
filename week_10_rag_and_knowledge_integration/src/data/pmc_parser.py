import xml.etree.ElementTree as ET 
from typing import Dict, List, Optional

def _get_text(elem: Optional[ET.Element]) -> str: 
    if elem is None: 
        return ''
    return ' '.join(''.join(elem.itertext()).split())

def _join_text(elems: List[ET.Element]) -> str: 
    return '\n\n'.join([_get_text(e) for e in elems if e is not None]).strip()

def parse_pmc_xml(xml_str: str) -> Dict: 
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
    results = []
    for x in xml_strings: 
        try: 
            results.append(parse_pmc_xml(x))
        except ET.ParseError: 
            pass
    return results
    