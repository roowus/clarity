from clarity.evidence import sentence_signals


def _base(**kw):
    args = dict(
        sent_score=0.95,
        sent_log_ppl=3.0,
        doc_score=0.9,
        threshold=0.85,
        neighbor_log_ppls=[3.0, 2.9, 3.1],
        sent_text="A perfectly ordinary sentence.",
        n_tokens=40,
    )
    args.update(kw)
    return args


def test_no_signals_for_clean_human_sentence():
    assert sentence_signals(**_base()) == []


def test_low_binoculars_fires():
    sigs = sentence_signals(**_base(sent_score=0.7))
    assert any(s.kind == "low_binoculars" for s in sigs)
    assert "0.700" in next(s for s in sigs if s.kind == "low_binoculars").detail


def test_low_burstiness_fires_on_flat_run():
    sigs = sentence_signals(
        **_base(neighbor_log_ppls=[2.0, 2.05, 2.02, 2.01], sent_log_ppl=2.0)
    )
    assert any(s.kind == "low_burstiness" for s in sigs)


def test_ai_idiom_detected_case_insensitive():
    sigs = sentence_signals(**_base(sent_text="Let us DELVE INTO the topic."))
    assert any(s.kind == "ai_idiom" for s in sigs)


def test_short_span_warns():
    sigs = sentence_signals(**_base(n_tokens=5))
    assert any(s.kind == "short_span" for s in sigs)
