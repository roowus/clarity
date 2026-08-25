from clairity.sentences import map_sentences_to_tokens, split_sentences


def test_split_covers_text():
    text = "First sentence here. Second one follows! And a third?"
    sents = split_sentences(text)
    assert len(sents) == 3
    assert sents[0][0].strip() == "First sentence here."


def test_token_mapping_partitions_tokens():
    text = "Cats sleep all day. Dogs bark at night."
    # Fake offsets: one "token" per word, char-accurate.
    offsets, pos = [], 0
    for word in text.split(" "):
        offsets.append((pos, pos + len(word)))
        pos += len(word) + 1
    spans = map_sentences_to_tokens(text, offsets)
    assert len(spans) == 2
    # Every token assigned exactly once, in order, no overlap.
    assert spans[0].tok_start == 0
    assert spans[0].tok_end == spans[1].tok_start
    assert spans[1].tok_end == len(offsets)
    assert spans[0].text.strip() == "Cats sleep all day."


def test_empty_sentences_dropped():
    text = "Hello."
    spans = map_sentences_to_tokens(text, [(0, 5), (5, 6)])
    assert len(spans) == 1
    assert spans[0].n_tokens == 2
