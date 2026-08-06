"""Tests for metadata filter evaluation."""

from app.knowledge.models import MetadataFilter
from app.knowledge.retrieval.filter import evaluate_filter, evaluate_filters


# ── evaluate_filter ───────────────────────────────────────────────────


class TestEvaluateFilterEq:
    def test_eq_match(self) -> None:
        f = MetadataFilter(field="status", op="eq", value="active")
        assert evaluate_filter({"status": "active"}, f) is True

    def test_eq_no_match(self) -> None:
        f = MetadataFilter(field="status", op="eq", value="active")
        assert evaluate_filter({"status": "deleted"}, f) is False

    def test_eq_missing_field(self) -> None:
        f = MetadataFilter(field="status", op="eq", value="active")
        assert evaluate_filter({}, f) is False


class TestEvaluateFilterNeq:
    def test_neq_match(self) -> None:
        f = MetadataFilter(field="status", op="neq", value="deleted")
        assert evaluate_filter({"status": "active"}, f) is True

    def test_neq_no_match(self) -> None:
        f = MetadataFilter(field="status", op="neq", value="active")
        assert evaluate_filter({"status": "active"}, f) is False


class TestEvaluateFilterIn:
    def test_in_match(self) -> None:
        f = MetadataFilter(field="type", op="in", value=["a", "b", "c"])
        assert evaluate_filter({"type": "b"}, f) is True

    def test_in_no_match(self) -> None:
        f = MetadataFilter(field="type", op="in", value=["a", "b", "c"])
        assert evaluate_filter({"type": "d"}, f) is False

    def test_in_non_iterable_value(self) -> None:
        f = MetadataFilter(field="type", op="in", value="scalar")
        assert evaluate_filter({"type": "scalar"}, f) is False


class TestEvaluateFilterNotIn:
    def test_not_in_match(self) -> None:
        f = MetadataFilter(field="type", op="not_in", value=["a", "b"])
        assert evaluate_filter({"type": "c"}, f) is True

    def test_not_in_no_match(self) -> None:
        f = MetadataFilter(field="type", op="not_in", value=["a", "b"])
        assert evaluate_filter({"type": "a"}, f) is False


class TestEvaluateFilterNumeric:
    def test_gt(self) -> None:
        f = MetadataFilter(field="score", op="gt", value=5)
        assert evaluate_filter({"score": 10}, f) is True
        assert evaluate_filter({"score": 5}, f) is False
        assert evaluate_filter({"score": 3}, f) is False

    def test_gte(self) -> None:
        f = MetadataFilter(field="score", op="gte", value=5)
        assert evaluate_filter({"score": 5}, f) is True
        assert evaluate_filter({"score": 4}, f) is False

    def test_lt(self) -> None:
        f = MetadataFilter(field="score", op="lt", value=5)
        assert evaluate_filter({"score": 3}, f) is True
        assert evaluate_filter({"score": 5}, f) is False

    def test_lte(self) -> None:
        f = MetadataFilter(field="score", op="lte", value=5)
        assert evaluate_filter({"score": 5}, f) is True
        assert evaluate_filter({"score": 6}, f) is False

    def test_numeric_with_none(self) -> None:
        f = MetadataFilter(field="score", op="gt", value=5)
        assert evaluate_filter({}, f) is False
        assert evaluate_filter({"score": None}, f) is False

    def test_numeric_type_error(self) -> None:
        f = MetadataFilter(field="score", op="gt", value=5)
        assert evaluate_filter({"score": "not_a_number"}, f) is False


class TestEvaluateFilterExists:
    def test_exists_true(self) -> None:
        f = MetadataFilter(field="page", op="exists", value=True)
        assert evaluate_filter({"page": 1}, f) is True

    def test_exists_false(self) -> None:
        f = MetadataFilter(field="page", op="exists", value=True)
        assert evaluate_filter({}, f) is False

    def test_not_exists(self) -> None:
        f = MetadataFilter(field="page", op="exists", value=False)
        assert evaluate_filter({}, f) is True


class TestEvaluateFilterContains:
    def test_contains_string(self) -> None:
        f = MetadataFilter(field="text", op="contains", value="hello")
        assert evaluate_filter({"text": "hello world"}, f) is True
        assert evaluate_filter({"text": "goodbye"}, f) is False

    def test_contains_list(self) -> None:
        f = MetadataFilter(field="tags", op="contains", value="python")
        assert evaluate_filter({"tags": ["python", "java"]}, f) is True
        assert evaluate_filter({"tags": ["java"]}, f) is False

    def test_contains_non_string_field(self) -> None:
        f = MetadataFilter(field="num", op="contains", value=5)
        assert evaluate_filter({"num": 42}, f) is False


class TestEvaluateFilterUnknown:
    def test_unknown_op(self) -> None:
        f = MetadataFilter(field="x", op="unknown_op", value=1)
        assert evaluate_filter({"x": 1}, f) is False


# ── evaluate_filters (conjunctive) ────────────────────────────────────


class TestEvaluateFilters:
    def test_empty_filters(self) -> None:
        assert evaluate_filters({"any": "data"}, ()) is True

    def test_all_pass(self) -> None:
        filters = (
            MetadataFilter(field="a", op="eq", value=1),
            MetadataFilter(field="b", op="eq", value=2),
        )
        assert evaluate_filters({"a": 1, "b": 2}, filters) is True

    def test_one_fails(self) -> None:
        filters = (
            MetadataFilter(field="a", op="eq", value=1),
            MetadataFilter(field="b", op="eq", value=2),
        )
        assert evaluate_filters({"a": 1, "b": 3}, filters) is False

    def test_namespace_filter(self) -> None:
        filters = (MetadataFilter(field="namespace", op="eq", value="prod"),)
        assert evaluate_filters({"namespace": "prod"}, filters) is True
        assert evaluate_filters({"namespace": "staging"}, filters) is False
