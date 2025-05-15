from tqdm import tqdm

def evaluate_spacy_ner(model, tagged_sents, label_map):
  """
  Evaluate a spaCy NER model against a BIO-tagged dataset.

  This function runs a spaCy NER model on a list of pre-tokenized sentences,
  converts the model's entity span predictions into BIO format, and aligns them 
  with gold-standard BIO labels for evaluation.

  Args:
      model (spacy.language.Language): A loaded spaCy NER pipeline (e.g., en_core_web_trf).
      tagged_sents (List[List[Tuple[str, str, str]]]): 
          A list of sentences, where each sentence is a list of (word, POS, entity) tuples.
      label_map (Dict[str, str]): 
          A mapping from spaCy's entity labels (e.g., 'PERSON', 'ORG') to the dataset's tag scheme (e.g., 'PER', 'ORG').

  Returns:
      Tuple[List[List[str]], List[List[str]]]: 
          - `predictions`: Model-predicted BIO labels per token.
          - `true_labels`: Gold-standard BIO labels from the dataset.

  Notes:
      - Assumes input sentences are already tokenized and follow the IOB format.
      - Uses spaCy's `char_span` with alignment_mode='expand' to map entity spans to tokens.
      - Entities not in `label_map` are defaulted to 'O'.
  """
  predictions, true_labels = [], []

  tqdm_bar = tqdm(total = len(tagged_sents), mininterval = 0)

  for s, sentence in enumerate(tagged_sents):
    words, _, entities = zip(*sentence)
    text = ' '.join(words)

    doc = model(text)
    preds = ['O'] * len(words)

    for ent in doc.ents:
      span = doc.char_span(ent.start_char, ent.end_char, alignment_mode = 'expand')

      if span:
        for i, word in enumerate(words):
          if text.find(word) == ent.start_char:
            prediction = label_map.get(ent.label_, 'O')
            preds[i] = f'B-{prediction}' if prediction != 'O' else 'O'
            for j in range(1, len(span)):
              if j + i < len(words):
                preds[j + i] = f'I-{prediction}' if prediction != 'O' else 'O'
            break

    predictions.append(preds)
    true_labels.append(list(entities))

    if s % 250 == 0:
      tqdm_bar.update(250)

  tqdm_bar.close()

  return predictions, true_labels

############################################################################################################
# Dictionary to map SpaCy labels to CoNLL labels 
############################################################################################################
label_map = {
    'PERSON': 'PER',
    'GPE': 'LOC',
    'ORG': 'ORG',
    'NORP': 'MISC',
    'FAC': 'MISC',
    'EVENT': 'MISC',
    'WORK_OF_ART': 'O',
    'LAW': 'O',
    'LANGUAGE': 'MISC',
    'PRODUCT': 'MISC',
    'DATE': 'O',
    'TIME': 'O',
    'PERCENT': 'O',
    'MONEY': 'O',
    'QUANTITY': 'O',
    'ORDINAL': 'O',
    'CARDINAL': 'O'
}