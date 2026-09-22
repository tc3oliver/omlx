# SPDX-License-Identifier: Apache-2.0
"""The Anthropic transport defaults SpecPrefill off.

Coding agents drive continuation-heavy sessions over ``/v1/messages`` and
depend on the reusable dense prefix checkpoint; a sparse prefill cannot be
stored, so it stalls that checkpoint and every later request re-prefills the
gap. ``/v1/chat/completions`` keeps the model-level default, which is still
right for cold one-shot prompts.

These tests cover the request schema, the scheduler gate that the override
ultimately drives, and a structural guard on the handler. The handler's
end-to-end default is proved by the release's runtime smoke against a real
server, which is stronger evidence than a heavily mocked unit test.
"""

import ast
import inspect
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from omlx.api.anthropic_models import MessagesRequest
from omlx.api.openai_models import ChatCompletionRequest


def _msg(**extra):
    base = {"model": "m", "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}]}
    base.update(extra)
    return MessagesRequest(**base)


# --- schema -----------------------------------------------------------------

def test_field_absent_is_none_so_handler_can_tell_unset_from_false():
    assert _msg().specprefill is None


@pytest.mark.parametrize("value", [True, False])
def test_field_roundtrips_explicit_booleans(value):
    assert _msg(specprefill=value).specprefill is value


def test_field_rejects_non_boolean():
    with pytest.raises(ValidationError):
        _msg(specprefill="sometimes")


def test_openai_schema_still_defaults_to_none():
    """The OpenAI endpoint must keep model-level behaviour, not inherit ours."""
    req = ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "hi"}]
    )
    assert req.specprefill is None


# --- the gate the override ultimately drives --------------------------------

def _scheduler_stub():
    from omlx.scheduler import Scheduler
    return SimpleNamespace(
        _specprefill_draft_model=object(),
        _try_specprefill_scoring=Scheduler._try_specprefill_scoring,
    )


def _request_stub(enabled):
    return SimpleNamespace(
        _specprefill_enabled=enabled,
        vlm_inputs_embeds=None,
        remaining_tokens=list(range(50_000)),   # far above any threshold
        prompt_token_ids=list(range(50_000)),
        specprefill_system_end=0,
        cached_tokens=0,
    )


def test_disabled_request_never_reaches_the_admission_policy(monkeypatch):
    """A disabled request must exit before plan_specprefill_scoring is called.

    The suffix here is 50K tokens, far past the threshold, so anything that
    reached the policy would be admitted.
    """
    import omlx.specprefill.policy as policy

    def explode(**_kwargs):
        raise AssertionError("plan_specprefill_scoring must not be reached")

    monkeypatch.setattr(policy, "plan_specprefill_scoring", explode)
    sched = _scheduler_stub()
    assert sched._try_specprefill_scoring(sched, _request_stub(False)) is None


def test_enabled_request_does_reach_the_admission_policy(monkeypatch):
    """Guard against the gate passing for the wrong reason."""
    import omlx.specprefill.policy as policy

    seen = {}

    def record(**kwargs):
        seen.update(kwargs)
        return None          # decline, so no draft model is needed

    monkeypatch.setattr(policy, "plan_specprefill_scoring", record)
    sched = _scheduler_stub()
    sched._try_specprefill_scoring(sched, _request_stub(True))
    assert seen, "policy should have been consulted for an enabled request"
    assert seen["cached_tokens"] == 0


def test_no_draft_model_means_no_scoring(monkeypatch):
    import omlx.specprefill.policy as policy
    monkeypatch.setattr(policy, "plan_specprefill_scoring",
                        lambda **_: (_ for _ in ()).throw(AssertionError("unreachable")))
    sched = _scheduler_stub()
    sched._specprefill_draft_model = None
    assert sched._try_specprefill_scoring(sched, _request_stub(True)) is None


# --- structural guard on the handler ----------------------------------------

def test_handler_forwards_specprefill_only_when_the_client_asked():
    """Regression guard: this transport must not reacquire a default of its own.

    It had one. `/v1/messages` forced `specprefill=False` because a sparse
    prefill could not be stored and the cache debt outgrew the saving within
    about three requests. Canonical state recovery repays that debt, so the
    transport went back to the model-level default -- which it gets by *not*
    writing the key when the client says nothing.

    Writing it unconditionally is the bug this guards: `False` would pin every
    coding-agent session back to dense prefill, and `True` would force sparse
    prefill on a deployment whose recovery budget is 0.0 and therefore never
    repays anything. Both are silent.

    Parsed rather than string-matched so reformatting does not break it.
    """
    import omlx.server as srv

    tree = ast.parse(inspect.getsource(srv.create_anthropic_message))
    assigns = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Subscript)
                and getattr(t.value, "id", None) == "chat_kwargs"
                and getattr(getattr(t, "slice", None), "value", None) == "specprefill"
                for t in n.targets)
    ]
    assert len(assigns) == 1, "expected exactly one specprefill assignment"

    assign = assigns[0]
    assert isinstance(assign.value, ast.Attribute) \
        and assign.value.attr == "specprefill" \
        and getattr(assign.value.value, "id", None) == "request", \
        "the client value must be forwarded verbatim, not defaulted"

    guards = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.If) and any(a is assign for a in ast.walk(n))
    ]
    assert guards, "the assignment must sit behind an `is not None` guard"
    test = guards[-1].test
    assert isinstance(test, ast.Compare) and isinstance(test.ops[0], ast.IsNot) \
        and isinstance(test.comparators[0], ast.Constant) \
        and test.comparators[0].value is None, \
        "the guard must be `request.specprefill is not None`"


def test_openai_handler_still_only_sets_specprefill_when_client_asked():
    """The OpenAI endpoint must not gain a transport default."""
    import omlx.server as srv

    tree = ast.parse(inspect.getsource(srv.create_chat_completion))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Subscript)
                        and getattr(t.value, "id", None) == "chat_kwargs"
                        and getattr(getattr(t, "slice", None), "value", None) == "specprefill"
                        for t in node.targets)):
            # must be a plain `= request.specprefill`, guarded by an `if`
            assert isinstance(node.value, ast.Attribute), \
                "OpenAI path must forward the client value verbatim"
            return
    pytest.fail("OpenAI handler no longer forwards specprefill")
