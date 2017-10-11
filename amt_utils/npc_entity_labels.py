import re


def convert_obj_pos_to_span(obj):
    td1 = test_vid.description()
    sent_lens = [len(sent.split()) + 1 for sent in td1.split('.')[:-1]]
    sent_n, word_n  = obj.data()['labelSpan']
    word_position = sum(sent_lens[:sent_n]) + word_n
    return (word_position, word_position + 1)

def check_overlap(span1, span2):
    x1, x2 = span1
    y1, y2 = span2
    return (x1 < y2) and (y1 < x2)

def get_char_spans(char):
    desc = td1
    td1.split()
    char_spans = [(m.start(), m.start() + len(char._data['entityLabel'].split())) for m in re.finditer(char._data['entityLabel'].lower(), desc.lower())]
    word_spans = compute_word_spans()
    return string_to_word_spans(char_spans[0], word_spans)

def compute_word_spans():
    words = td1.split()
    word_spans = []
    for idx, word in enumerate(words):
        if word_spans:
            last_idx = word_spans[-1][1]
            word_spans.append((last_idx, last_idx + 1 + len(word)))
        else: 
            word_spans.append((0, len(word) + 1))
    return word_spans

def string_to_word_spans(match_span, word_spans):
    spans = [idx for idx, word_span in enumerate(word_spans) if check_overlap(word_span, match_span)]
    last_seen = []
    for idx, word_idx in enumerate(spans):
        if idx == 0:
            last_seen.append(word_idx)
        elif word_idx + 1 == last_seen[-1]:
            last_seen.append(word_idx)
    return spans[0], spans[-1]
    
def assign_char_npcs(entites, comp_chars=True):
    characters = test_vid._data['characters']
    objects = test_vid._data['characters']
    chunk_spans = test_vid._data['parse']['noun_phrase_chunks']['chunks']
    chunk_names = test_vid._data['parse']['noun_phrase_chunks']['named_chunks']
    for ent in entites:
        if ent._data['entityLabel'].lower() in main_characters_lower:
            continue
        if comp_chars:
            ent_spans = get_char_spans(ent)
        else:
            ent_spans = convert_obj_pos_to_span(ent)

        for idx, chunk_span in enumerate(chunk_spans):
            if check_overlap(ent_spans, chunk_span):
                ent._data['labelNPC'] = chunk_names[idx]
