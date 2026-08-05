"""Tests for the injectable clock abstraction."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest


def test_system_clock_returns_aware_utc(fake_clock, system_clock) -> None:
    now = system_clock.utcnow()
    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(0)


def test_fake_clock_starts_at_fixed_timestamp(fake_clock) -> None:
    assert fake_clock.utcnow() == datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_fake_clock_advance(fake_clock) -> None:
    fake_clock.advance(90.5)
    expected = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=90.5)
    assert fake_clock.utcnow() == expected
    assert fake_clock.time() == pytest.approx(90.5)


def test_fake_clock_set(fake_clock) -> None:
    target = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
    fake_clock.set(target)
    assert fake_clock.utcnow() == target


def test_fake_clock_normalises_naive_datetimes(fake_clock) -> None:
    target = datetime(2026, 3, 15, 12, 0, 0)
    fake_clock.set(target)
    assert fake_clock.utcnow().tzinfo is not None


def test_fake_clock_advance_updates_monotonic(fake_clock) -> None:
    fake_clock.advance(5)
    fake_clock.advance(7)
    assert fake_clock.time() == pytest.approx(12.0)
