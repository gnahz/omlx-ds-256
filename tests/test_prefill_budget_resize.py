"""Pure byte-ledger tests: resizing never allocates a model or touches MLX."""
from copy import deepcopy
from dataclasses import replace

import pytest

from omlx.request_memory_budget import (
    GIB, RequestMemoryBudget, RequestMemoryEnvelope, physical_headroom_budget,
)


MIB = 1024**2


def envelope(workspace, *, resident=0, credit_cap=0, chunk=256, decode=0):
    return RequestMemoryEnvelope(
        context_tokens=371040,
        resident_bytes=resident,
        prefill_bytes=workspace,
        decode_bytes=resident + decode,
        total_bytes=resident + max(workspace, decode),
        credit_cap_bytes=credit_cap,
        prefill_chunk_tokens=chunk,
    )


def state(budget):
    return deepcopy((budget.reservations, budget._materialized, budget.decoding,
                     budget.shared_reclaim_bytes, budget.last_decision))


def test_upgrade_replaces_instead_of_adding_both_plans():
    b = RequestMemoryBudget()
    old, new = envelope(4*GIB), envelope(6*GIB, chunk=512)
    assert b.reserve('prefill', old, 230*GIB, 240*GIB)
    admission = deepcopy(b.last_decision)

    assert b.try_resize('prefill', new, 230*GIB, 240*GIB)

    assert b.reservations['prefill'] is new
    assert b.reserved_bytes() == 6*GIB
    assert b.last_decision == admission
    assert b.last_resize_decision['predicted_peak_bytes'] == 236*GIB
    assert b.last_resize_decision['previous_chunk_tokens'] == 256
    assert b.last_resize_decision['selected_chunk_tokens'] == 512


def test_failed_upgrade_retains_all_ownership_and_queue_head_diagnostics():
    b = RequestMemoryBudget()
    old = envelope(2*GIB, resident=3*GIB, credit_cap=GIB)
    b.reservations['prefill'] = old
    b.note_materialized('prefill', {111: GIB})
    b.shared_reclaim_bytes = GIB
    b.reservations['old-decode'] = envelope(GIB, decode=GIB)
    b.mark_decoding('old-decode')
    b.last_decision = {'request_id': 'waiting-head', 'allowed': False}
    before = state(b)
    materialized = b._materialized['prefill']

    assert not b.try_resize('prefill', replace(old, prefill_bytes=5*GIB,
                                             total_bytes=8*GIB, prefill_chunk_tokens=1024),
                            230*GIB, 240*GIB)

    assert state(b) == before
    assert b.reservations['prefill'] is old
    assert b._materialized['prefill'] is materialized
    assert not b.last_resize_decision['allowed']


def test_successful_upgrade_keeps_confirmed_kv_credit_and_release_lifecycle():
    b = RequestMemoryBudget()
    old = envelope(GIB, resident=6*GIB, credit_cap=2*GIB)
    b.reservations['prefill'] = old
    b.note_materialized('prefill', {111: 2*GIB})
    new = replace(old, prefill_bytes=2*GIB, total_bytes=8*GIB, prefill_chunk_tokens=512)
    materialized = b._materialized['prefill']

    assert b.try_resize('prefill', new, 232*GIB, 240*GIB)

    assert b._materialized['prefill'] is materialized
    assert b._credits()['prefill'] == 2*GIB
    assert b.reserved_bytes() == 6*GIB
    assert b.last_resize_decision['predicted_peak_bytes'] == 238*GIB
    b.release('prefill')
    assert b.reservations == b._materialized == {}


def test_shared_kv_is_credited_once_across_candidate_and_other_request():
    b = RequestMemoryBudget()
    old = envelope(GIB, resident=6*GIB, credit_cap=2*GIB)
    b.reservations = {'first': old, 'second': old}
    for rid in b.reservations:
        b.note_materialized(rid, {111: 2*GIB})
    new = replace(old, prefill_bytes=2*GIB, total_bytes=8*GIB, prefill_chunk_tokens=512)
    before = state(b)

    # Correct charge is 8 + 7 - 2 = 13 GiB. Crediting both owners
    # would incorrectly accept this at 237 GiB.
    assert not b.try_resize('first', new, 226*GIB, 240*GIB)
    assert b.last_resize_decision['reserved_bytes'] == 13*GIB
    assert state(b) == before
    assert b.try_resize('first', new, 225*GIB, 240*GIB)
    assert list(b.reservations) == ['first', 'second']
    assert sum(b._credits().values()) == 2*GIB


def test_candidate_credit_cap_is_used_without_claiming_unconfirmed_buffers():
    b = RequestMemoryBudget()
    old = envelope(GIB, resident=3*GIB, credit_cap=GIB)
    b.reservations['prefill'] = old
    b.note_materialized('prefill', {111: 2*GIB})
    new = replace(old, resident_bytes=6*GIB, credit_cap_bytes=2*GIB,
                  prefill_bytes=2*GIB, total_bytes=8*GIB, prefill_chunk_tokens=512)

    assert b.try_resize('prefill', new, 232*GIB, 240*GIB)
    assert b._credits()['prefill'] == 2*GIB
    b.note_materialized('prefill', {})
    before = state(b)
    assert not b.try_resize('prefill', new, 232*GIB, 240*GIB)
    assert state(b) == before


@pytest.mark.parametrize('separate', [False, True])
def test_other_decode_plan_and_shared_workspace_are_charged_once(separate):
    b = RequestMemoryBudget()
    b.separate_decode_workspace = separate
    b.shared_reclaim_bytes = GIB
    b.reservations['prefill'] = envelope(GIB)
    b.reservations['old-decode'] = envelope(4*GIB, decode=GIB)
    b.mark_decoding('old-decode')

    assert b.try_resize('prefill', envelope(3*GIB, chunk=512), 233*GIB, 240*GIB)

    # New prefill + decode verify + a single shared reservation.
    assert b.reserved_bytes() == (3 + 1 + 1)*GIB
    assert b.last_resize_decision['predicted_peak_bytes'] == 238*GIB
    assert b.decoding == {'old-decode'}


@pytest.mark.parametrize('reason', ['missing_reservation', 'already_decoding'])
def test_resize_cannot_create_or_take_over_a_decode_reservation(reason):
    b = RequestMemoryBudget()
    if reason == 'already_decoding':
        b.reservations['target'] = envelope(GIB, decode=GIB)
        b.mark_decoding('target')
        b.note_materialized('target', {111: 123})
    before = state(b)

    assert not b.try_resize('target', envelope(2*GIB, chunk=512), 220*GIB, 240*GIB)

    assert state(b) == before
    assert b.last_resize_decision == {'request_id': 'target', 'allowed': False, 'reason': reason}


@pytest.mark.parametrize('limit', [0, -1, 236*GIB, 240*GIB, 300*GIB])
def test_resize_respects_unavailable_tighter_and_absolute_admission_limits(limit):
    b = RequestMemoryBudget()
    b.reservations['prefill'] = envelope(GIB)
    target = envelope(2*GIB, chunk=512)
    expected = limit >= 238*GIB

    assert b.try_resize('prefill', target, 236*GIB, limit) is expected
    assert b.last_resize_decision['limit_bytes'] == (min(limit, 238*GIB) if limit > 0 else 0)
    assert not b.try_resize('prefill', target, 237*GIB, limit)


def test_downsize_can_restore_fit_but_cannot_bypass_current_physical_usage():
    b = RequestMemoryBudget()
    old = envelope(4*GIB, chunk=1024)
    b.reservations['prefill'] = old
    smaller = envelope(2*GIB, chunk=256)

    assert b.try_resize('prefill', smaller, 236*GIB, 240*GIB)
    assert b.reserved_bytes() == 2*GIB
    before = state(b)
    assert not b.try_resize('prefill', smaller, 237*GIB, 240*GIB)
    assert state(b) == before


def test_resize_spends_released_ram_without_fixed_reserve_and_ignores_speculative_ram():
    b = RequestMemoryBudget()
    b.reservations['prefill'] = envelope(GIB)
    current = 222*GIB
    new = envelope(2*GIB, chunk=512)
    low = physical_headroom_budget(current, {'free': 2*GIB, 'speculative': GIB})
    assert not b.try_resize('prefill', new, current, low['system_limit_bytes'])
    enough = physical_headroom_budget(current, {'free': 2*GIB, 'speculative': 0})
    assert b.try_resize('prefill', new, current, enough['system_limit_bytes'])
    assert enough['system_free_reserve_bytes'] == 0


def test_upgrade_margin_is_only_a_fit_requirement_not_a_persistent_charge():
    b = RequestMemoryBudget()
    b.reservations['prefill'] = envelope(GIB)
    margin = 128*MIB
    new = envelope(8*GIB - margin, chunk=512)
    before = state(b)

    assert not b.try_resize('prefill', new, 230*GIB + 1, 240*GIB,
                            headroom_margin=margin)
    assert state(b) == before
    assert b.try_resize('prefill', new, 230*GIB, 240*GIB,
                        headroom_margin=margin)
    assert b.reserved_bytes() == 8*GIB - margin
    assert b.last_resize_decision['predicted_peak_bytes'] == 238*GIB - margin
    assert b.snapshot(230*GIB, 240*GIB)['available_bytes'] == margin


def test_negative_margin_cannot_create_budget_headroom():
    b = RequestMemoryBudget()
    b.reservations['prefill'] = envelope(GIB)
    assert not b.try_resize('prefill', envelope(2*GIB, chunk=512),
                            237*GIB, 240*GIB, headroom_margin=-GIB)
    assert b.last_resize_decision['headroom_margin_bytes'] == 0


def test_legacy_credit_limit_remains_one_confirmed_resident_copy():
    b = RequestMemoryBudget()
    old = envelope(GIB, resident=6*GIB, credit_cap=None)
    b.reservations['prefill'] = old
    b.note_materialized('prefill', {111: 4*GIB})
    new = replace(old, prefill_bytes=2*GIB, total_bytes=8*GIB, prefill_chunk_tokens=512)
    assert b.try_resize('prefill', new, 232*GIB, 240*GIB)
    assert b.reserved_bytes() == 6*GIB
