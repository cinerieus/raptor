"""Shared verdict-vote tally — abstention semantics across all voters.

One counting rule (``correlation.tally_verdict_votes``) backs every
surface that counts ``is_exploitable`` votes: the correlation engine,
``ConsensusTask.finalize``, and ``JudgeTask.finalize``. A missing or
null ``is_exploitable`` (errored model, refused response, schema
failure nulled by response validation) is an ABSTENTION — it must
never count as a "not exploitable" vote, and a stage whose whole
panel abstained must report an explicit no-verdict outcome instead of
minting agreement.
"""

from packages.llm_analysis.correlation import (
    VoteTally,
    tally_verdict_votes,
)
from packages.llm_analysis.tasks import ConsensusTask, JudgeTask


class TestTallyVerdictVotes:
    def test_counts_each_bucket(self):
        t = tally_verdict_votes([True, False, None, True, None])
        assert t == VoteTally(exploitable=2, not_exploitable=1, abstained=2)
        assert t.voted == 3

    def test_majority_is_strict_over_voters_only(self):
        # 1 yes + 2 abstains: the single real vote IS the majority —
        # pre-fix the abstainers read as 2 "no" votes and inverted it.
        assert tally_verdict_votes([True, None, None]).majority() is True
        assert tally_verdict_votes([False, None, None]).majority() is False
        assert tally_verdict_votes([True, False, False]).majority() is False

    def test_tie_has_no_majority(self):
        t = tally_verdict_votes([True, False])
        assert t.tie is True
        assert t.majority() is None

    def test_all_abstain_is_explicitly_inconclusive(self):
        t = tally_verdict_votes([None, None, None])
        assert t.voted == 0
        assert t.majority() is None
        assert t.tie is False
        assert t.disputed is False
        assert t.unanimous is False

    def test_abstainers_never_create_a_dispute(self):
        assert tally_verdict_votes([True, None]).disputed is False
        assert tally_verdict_votes([True, None]).unanimous is True
        assert tally_verdict_votes([True, False]).disputed is True

    def test_truthy_values_are_boolean_coerced(self):
        t = tally_verdict_votes([1, 0, "yes"])
        assert t.exploitable == 2
        assert t.not_exploitable == 1


def _consensus(fid: str, is_exploitable, model: str = "m2") -> dict:
    return {
        "finding_id": fid,
        "analysed_by": model,
        "is_exploitable": is_exploitable,
        "reasoning": "r",
    }


class TestConsensusFinalizeAbstention:
    def test_two_abstaining_models_do_not_flip_exploitable(self):
        # The verdict-direction hazard: a real "exploitable" primary
        # plus two consensus responses whose verdicts were nulled by
        # response validation. Pre-fix the abstainers counted as two
        # False votes — a 2-1 "majority" AGAINST — and the true
        # positive was silently downgraded.
        primary = {"is_exploitable": True}
        results = [_consensus("f1", None, "m2"), _consensus("f1", None, "m3")]
        ConsensusTask().finalize(results, {"f1": primary})
        assert primary["is_exploitable"] is True
        assert primary["consensus"] == "no-verdict"

    def test_abstainer_plus_real_vote_uses_only_the_real_vote(self):
        primary = {"is_exploitable": True}
        results = [_consensus("f1", True, "m2"), _consensus("f1", None, "m3")]
        ConsensusTask().finalize(results, {"f1": primary})
        assert primary["is_exploitable"] is True
        assert primary["consensus"] == "agreed"

    def test_genuine_not_exploitable_majority_still_wins(self):
        primary = {"is_exploitable": True}
        results = [
            _consensus("f1", False, "m2"),
            _consensus("f1", False, "m3"),
        ]
        ConsensusTask().finalize(results, {"f1": primary})
        assert primary["is_exploitable"] is False
        assert primary["consensus"] == "disputed"

    def test_all_abstain_leaves_primary_untouched(self):
        primary = {"is_exploitable": False}
        results = [_consensus("f1", None, "m2"), _consensus("f1", None, "m3")]
        ConsensusTask().finalize(results, {"f1": primary})
        assert primary["is_exploitable"] is False
        assert primary["consensus"] == "no-verdict"
        # The per-model record still lands for operator inspection.
        assert len(primary["consensus_analyses"]) == 2

    def test_tie_resolves_conservative_exploitable(self):
        # primary True + one False + one abstainer = a 1-1 tie among
        # actual voters: no majority — conservative-max applies (same
        # rule as the 1-vote dispute), surfacing the finding for
        # review instead of silently resolving against it.
        primary = {"is_exploitable": True}
        results = [
            _consensus("f1", False, "m2"),
            _consensus("f1", None, "m3"),
        ]
        ConsensusTask().finalize(results, {"f1": primary})
        assert primary["is_exploitable"] is True
        assert primary["consensus"] == "disputed"

    def test_single_consensus_conservative_max_unchanged(self):
        # Pre-existing 1-vote rule survives the tally refactor.
        primary = {"is_exploitable": False}
        results = [_consensus("f1", True, "m2")]
        ConsensusTask().finalize(results, {"f1": primary})
        assert primary["is_exploitable"] is True
        assert primary["consensus"] == "disputed"

    def test_pre_consensus_verdict_still_captured(self):
        primary = {"is_exploitable": False}
        results = [_consensus("f1", True, "m2")]
        ConsensusTask().finalize(results, {"f1": primary})
        assert primary["pre_consensus_is_exploitable"] is False


def _judge(fid: str, is_exploitable, model: str = "j1") -> dict:
    return {
        "finding_id": fid,
        "analysed_by": model,
        "is_exploitable": is_exploitable,
        "reasoning": "r",
    }


class TestJudgeFinalizeAbstention:
    def test_abstaining_judge_cannot_flip_majority(self):
        # primary True + judges [True, None, False]: 2-1 among actual
        # voters keeps True. Pre-fix the abstainer was a False vote —
        # 2-2, "no majority" resolved to False.
        primary = {"is_exploitable": True}
        results = [
            _judge("f1", True, "j1"),
            _judge("f1", None, "j2"),
            _judge("f1", False, "j3"),
        ]
        JudgeTask().finalize(results, {"f1": primary})
        assert primary["is_exploitable"] is True
        assert primary["judge"] == "disputed"

    def test_genuine_judge_majority_against_still_wins(self):
        primary = {"is_exploitable": True}
        results = [_judge("f1", False, "j1"), _judge("f1", False, "j2")]
        JudgeTask().finalize(results, {"f1": primary})
        assert primary["is_exploitable"] is False

    def test_all_judges_abstain_preserves_primary(self):
        primary = {"is_exploitable": True, "self_contradictory": True}
        results = [_judge("f1", None, "j1"), _judge("f1", None, "j2")]
        JudgeTask().finalize(results, {"f1": primary})
        assert primary["is_exploitable"] is True
        assert primary["judge"] == "no-verdict"
        # No judge voted: nothing exists to tie-break a
        # self-contradiction with — the flag must survive.
        assert primary["self_contradictory"] is True
        assert "contradiction_resolved_by_judge" not in primary
        assert len(primary["judge_analyses"]) == 2

    def test_tie_preserves_primary(self):
        # Judge overrides only on a strict majority.
        primary = {"is_exploitable": True}
        results = [_judge("f1", True, "j1"), _judge("f1", False, "j2")]
        JudgeTask().finalize(results, {"f1": primary})
        assert primary["is_exploitable"] is True

    def test_voting_judge_still_resolves_contradiction(self):
        primary = {"is_exploitable": True, "self_contradictory": True}
        results = [_judge("f1", True, "j1")]
        JudgeTask().finalize(results, {"f1": primary})
        assert primary["self_contradictory"] is False
        assert primary["contradiction_resolved_by_judge"] is True

    def test_single_judge_preserves_primary_verdict(self):
        primary = {"is_exploitable": False}
        results = [_judge("f1", True, "j1")]
        JudgeTask().finalize(results, {"f1": primary})
        assert primary["is_exploitable"] is False
        assert primary["judge"] == "disputed"
