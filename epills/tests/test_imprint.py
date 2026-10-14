"""Imprint matching and re-ranking (epills.core.imprint.match); the OCR model itself isn't loaded here."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from epills.core.imprint.match import ImprintIndex, imprint_tokens, rerank  # noqa: E402

IMPRINTS = ["TEVA;5729", "93;3145", "93;7146", "93;1234", "GG;738", "", "M;547"]


def test_tokens():
    assert imprint_tokens("TEVA;5729") == {"TEVA", "5729"}
    assert imprint_tokens("M;547") == {"547"}  # single characters carry too little evidence
    assert imprint_tokens("") == set()


def test_rare_tokens_weigh_more_and_merged_reads_count():
    index = ImprintIndex(IMPRINTS)
    assert index.weight["93"] < index.weight["3145"]  # '93' is on three pills
    s = index.scores(["", "738GG", "x"], [4, 0])        # 'GG' and '738' read run together
    assert s[0] > 0 and s[1] == 0
    assert index.scores(["93"], [1])[0] < index.scores(["3145"], [1])[0]
    assert (index.scores([], [0, 1]) == 0).all() and index.scores(["TEVA"], [5])[0] == 0  # nothing read / unknown


def test_rerank_moves_the_imprint_match_up_only_among_the_top_k():
    index = ImprintIndex(IMPRINTS)
    sims = np.array([[0.90, 0.85, 0.80, 0.70, 0.60, 0.50, 0.40]], np.float32)
    out = rerank(sims, [["93 3145"]], index, alpha=0.3, top_k=3)
    assert out[0].argmax() == 1                          # the read '3145' lifts candidate 1 above candidate 0
    assert np.allclose(out[0, 3:], sims[0, 3:])          # outside the top 3, nothing changes
    assert np.allclose(rerank(sims, [[]], index, 0.3), sims)  # nothing read: the embedding's order stays


def test_read_hits_weighs_tokens_and_ignores_nothing_read():
    from epills.core.imprint.match import read_hits
    imprints = {"m1": "TEVA;5729", "m2": "93;3145", "m3": ""}
    weight = ImprintIndex(IMPRINTS).weight
    assert set(read_hits(["", "5729 TEVA"], imprints, weight)) == {"m1"}
    assert read_hits(["93"], imprints, weight, min_weight=0.9) == {}  # '93' is common: too weak alone
    assert read_hits(["", ""], imprints, weight) == {}


def test_only_tokens_that_tell_the_gallery_apart_count_when_distinct():
    from epills.core.imprint.match import distinct_tokens, read_hits
    gallery = {"tramadol": "TV;58", "carvedilol": "TV;7295", "other": "72;AN"}  # blueprint B27: same maker, other drug
    assert distinct_tokens(gallery) == {"tramadol": {"58"}, "carvedilol": {"7295"}, "other": {"AN"}}  # '72' in '7295'
    assert set(read_hits(["TV 58"], gallery)) == {"tramadol", "carvedilol"}  # 'TV' counts for both
    assert set(read_hits(["TV 58"], gallery, distinct=True)) == {"tramadol"}
    assert read_hits(["TV"], gallery, distinct=True) == {}


def test_imprint_adapter_wraps_text_layers_switches_off_and_reloads(tmp_path):
    import torch
    from torch import nn
    from epills.core.imprint.lora import add_lora, load_lora, lora_state, set_lora

    class Tiny(nn.Module):  # named like Florence-2: model.vision_tower / model.language_model
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.model = nn.Module()
            self.model.vision_tower, self.model.language_model = nn.Module(), nn.Module()
            self.model.vision_tower.qkv, self.model.language_model.q_proj = nn.Linear(4, 4), nn.Linear(4, 4)

        def forward(self, x):
            return self.model.language_model.q_proj(self.model.vision_tower.qkv(x))

    x, net = torch.randn(2, 4), Tiny()
    base = net(x)
    assert add_lora(net, r=2, alpha=4) == ["model.language_model.q_proj"]  # the vision tower stays as it is
    with torch.no_grad():
        net.model.language_model.q_proj.lora_b.fill_(0.5)
    assert not torch.allclose(net(x), base)
    set_lora(net, False)
    assert torch.allclose(net(x), base)  # off: exactly the original model (labels, captions)
    set_lora(net, True)
    torch.save({"config": {"r": 2, "alpha": 4, "name": "t"}, "state": lora_state(net)}, tmp_path / "a.pt")
    again = Tiny()
    assert load_lora(again, tmp_path / "a.pt")["name"] == "t" and torch.allclose(again(x), net(x))


def test_with_distinct_tokens_a_shared_token_never_confirms_and_never_accepts_more():
    from epills.core.inference.matching import Decision, check_imprint
    wrong = Decision("carvedilol", 0.95, "tramadol", 0.70, "match")  # a tramadol pill that looks like carvedilol
    both = {"tramadol", "carvedilol"}
    # read 'TV 58': before, 'TV' confirmed carvedilol; with distinct tokens only '58' counts, so it's a conflict
    assert check_imprint(wrong, both, both, 0.9)[1] == "confirmed"
    d, status = check_imprint(wrong, both, both, 0.9, distinct_hits={"tramadol"})
    assert (d.decision, status) == ("ambiguous", "conflict")
    assert check_imprint(wrong, both, both, 0.9, distinct_hits=set())[1] == "unresolved"  # only 'TV' read
    # whatever was read, the distinct check never keeps a match the plain check turned down
    for hits in ({"tramadol"}, {"carvedilol"}, both):
        for dh in (set(), {"tramadol"}, {"carvedilol"}, both):
            if dh <= hits and check_imprint(wrong, hits, both, 0.9)[0].decision == "ambiguous":
                assert check_imprint(wrong, hits, both, 0.9, dh)[0].decision == "ambiguous"


def test_check_imprint_confirms_flags_and_resolves_but_never_accepts_an_unknown_pill():
    from epills.core.inference.matching import Decision, check_imprint
    match = Decision("m1", 0.95, "m2", 0.70, "match")
    tie = Decision("m1", 0.93, "m2", 0.91, "ambiguous")
    unknown = Decision("m1", 0.60, "m2", 0.50, "unknown")
    both = {"m1", "m2"}
    assert check_imprint(match, {"m1"}, both, 0.9)[1] == "confirmed"
    d, status = check_imprint(match, {"m2"}, both, 0.9)
    assert (d.decision, d.medication_id, status) == ("ambiguous", "m1", "conflict")
    d, status = check_imprint(tie, {"m2"}, both, 0.9)
    assert (d.decision, d.medication_id, d.runner_up_id, status) == ("match", "m2", "m1", "resolved")
    assert check_imprint(tie, {"m2"}, both, 0.95)[0].decision == "ambiguous"   # runner-up below the threshold
    assert check_imprint(tie, both, both, 0.9)[1] == "unresolved"
    assert check_imprint(unknown, {"m1"}, both, 0.9) == (unknown, "not_used")
    assert check_imprint(match, set(), both, 0.9) == (match, "not_read")
    assert check_imprint(match, set(), set(), 0.9) == (match, "not_checked")
