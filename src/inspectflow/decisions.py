"""Map inspection signals to an ERP quality decision."""

from __future__ import annotations

from inspectflow.schemas import Decision


def decide(
    *,
    counted: int,
    expected_quantity: int | None,
    defect_count: int,
    work_order_matches: bool | None,
    operator_verified: bool | None,
) -> Decision:
    """Reject scrap, and hold the lot for a person when the count or identity disagrees.

    `None` means that signal was not supplied, so it does not block the lot.
    """
    if counted < 0 or defect_count < 0:
        raise ValueError("counts cannot be negative")
    if defect_count > 0:
        return Decision.reject
    quantity_gap = expected_quantity is not None and counted != expected_quantity
    identity_gap = work_order_matches is False or operator_verified is False
    if quantity_gap or identity_gap:
        return Decision.review
    return Decision.accept
