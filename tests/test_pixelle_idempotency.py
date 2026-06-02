#!/usr/bin/env python3
"""
Tests for Pixelle Idempotency Key Generation and Dedupe Store Interface

Test coverage:
- Idempotency key generation determinism and collision resistance
- Payload normalization and volatile field exclusion
- Dedupe store interface (InMemoryDedupeStore)
- Job handle lifecycle and get_or_create semantics
- Duplicate request detection and existing job return
- Changed payload/workflow produces distinct key
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from pixelle_snapshot.idempotency import (
    compute_idempotency_key,
    compute_request_fingerprint,
    JobHandle,
    InMemoryDedupeStore,
    create_dedupe_store,
    get_default_store,
    _normalize_payload,
    _hash_dict,
)


# ─────────────────────────────────────────────
# Idempotency Key Generation Tests
# ─────────────────────────────────────────────
class TestIdempotencyKeyGeneration:
    def test_same_inputs_produce_same_key(self):
        """Same segment/workflow/payload always produces identical key."""
        payload = {"input_image_path": "/path/to/image.png", "motion_type": "kenburns"}
        
        key1 = compute_idempotency_key("seg123#1", "i2v", payload)
        key2 = compute_idempotency_key("seg123#1", "i2v", payload)
        
        assert key1 == key2
        assert len(key1) == 32  # 32-char hex string

    def test_different_segment_produces_different_key(self):
        """Different segment_key produces distinct idempotency key."""
        payload = {"input_image_path": "/path/to/image.png"}
        
        key1 = compute_idempotency_key("seg123#1", "i2v", payload)
        key2 = compute_idempotency_key("seg456#1", "i2v", payload)
        
        assert key1 != key2

    def test_different_workflow_produces_different_key(self):
        """Different workflow produces distinct idempotency key."""
        payload = {"input_image_path": "/path/to/image.png"}
        
        key1 = compute_idempotency_key("seg123#1", "i2v", payload)
        key2 = compute_idempotency_key("seg123#1", "digital_human", payload)
        
        assert key1 != key2

    def test_different_payload_produces_different_key(self):
        """Changed payload produces distinct idempotency key."""
        key1 = compute_idempotency_key(
            "seg123#1", "i2v",
            {"input_image_path": "/path/to/image1.png", "motion_type": "kenburns"}
        )
        key2 = compute_idempotency_key(
            "seg123#1", "i2v",
            {"input_image_path": "/path/to/image2.png", "motion_type": "kenburns"}
        )
        
        assert key1 != key2

    def test_payload_order_independence(self):
        """Payload field order does not affect key."""
        payload1 = {"a": 1, "b": 2, "c": 3}
        payload2 = {"c": 3, "b": 2, "a": 1}
        
        key1 = compute_idempotency_key("seg#1", "i2v", payload1)
        key2 = compute_idempotency_key("seg#1", "i2v", payload2)
        
        assert key1 == key2

    def test_empty_segment_key_raises(self):
        """Empty segment_key raises ValueError."""
        with pytest.raises(ValueError, match="segment_key is required"):
            compute_idempotency_key("", "i2v", {})

    def test_empty_workflow_raises(self):
        """Empty workflow raises ValueError."""
        with pytest.raises(ValueError, match="workflow is required"):
            compute_idempotency_key("seg#1", "", {})


class TestPayloadNormalization:
    def test_volatile_fields_excluded(self):
        """Volatile fields are excluded from hash."""
        payload1 = {
            "input_image_path": "/path/to/image.png",
            "metadata": {"request_id": "abc123", "timestamp": "2026-03-15"},
            "request_id": "xyz789",
            "timestamp": "2026-03-15T10:00:00Z",
            "created_at": "2026-03-15T10:00:00Z",
            "timeout_seconds": 300.0,
        }
        payload2 = {
            "input_image_path": "/path/to/image.png",
            "metadata": {"request_id": "different", "timestamp": "2026-03-16"},
            "request_id": "different_id",
            "timestamp": "2026-03-16T11:00:00Z",
            "created_at": "2026-03-16T11:00:00Z",
            "timeout_seconds": 600.0,
        }
        
        key1 = compute_idempotency_key("seg#1", "i2v", payload1)
        key2 = compute_idempotency_key("seg#1", "i2v", payload2)
        
        assert key1 == key2

    def test_none_values_excluded(self):
        """None values are excluded from hash."""
        payload1 = {"a": 1, "b": None}
        payload2 = {"a": 1}
        
        norm1 = _normalize_payload(payload1)
        norm2 = _normalize_payload(payload2)
        
        assert norm1 == norm2

    def test_nested_dict_normalized(self):
        """Nested dicts are recursively normalized."""
        payload = {
            "outer": {
                "inner": "value",
                "metadata": {"should": "be_excluded"},
            }
        }
        
        normalized = _normalize_payload(payload)
        
        assert "metadata" not in normalized["outer"]
        assert normalized["outer"]["inner"] == "value"


class TestRequestFingerprint:
    def test_fingerprint_from_dict(self):
        """Fingerprint computed from dict payload."""
        payload = {"input_image_path": "/path/to/image.png"}
        
        fp = compute_request_fingerprint(payload)
        
        assert len(fp) == 16
        assert fp == compute_request_fingerprint(payload)  # Deterministic

    def test_fingerprint_ignores_volatile(self):
        """Fingerprint excludes volatile fields."""
        payload1 = {"input_image_path": "/path/to/image.png", "metadata": {"foo": "bar"}}
        payload2 = {"input_image_path": "/path/to/image.png"}
        
        fp1 = compute_request_fingerprint(payload1)
        fp2 = compute_request_fingerprint(payload2)
        
        assert fp1 == fp2


# ─────────────────────────────────────────────
# Job Handle Tests
# ─────────────────────────────────────────────
class TestJobHandle:
    def test_handle_creation_with_defaults(self):
        """JobHandle created with sensible defaults."""
        handle = JobHandle(job_id="job123", idempotency_key="key456")
        
        assert handle.job_id == "job123"
        assert handle.idempotency_key == "key456"
        assert handle.status == "submitted"
        assert handle.created_at.endswith("Z")
        assert handle.updated_at == handle.created_at
        assert handle.output_path is None
        assert handle.error_code is None

    def test_is_active_states(self):
        """is_active() returns True for non-terminal states."""
        submitted = JobHandle(job_id="j1", idempotency_key="k1", status="submitted")
        queued = JobHandle(job_id="j2", idempotency_key="k2", status="queued")
        running = JobHandle(job_id="j3", idempotency_key="k3", status="running")
        succeeded = JobHandle(job_id="j4", idempotency_key="k4", status="succeeded")
        failed = JobHandle(job_id="j5", idempotency_key="k5", status="failed")
        
        assert submitted.is_active() is True
        assert queued.is_active() is True
        assert running.is_active() is True
        assert succeeded.is_active() is False
        assert failed.is_active() is False

    def test_is_succeeded(self):
        """is_succeeded() returns True only for succeeded status."""
        succeeded = JobHandle(job_id="j1", idempotency_key="k1", status="succeeded")
        failed = JobHandle(job_id="j2", idempotency_key="k2", status="failed")
        
        assert succeeded.is_succeeded() is True
        assert failed.is_succeeded() is False

    def test_to_dict_and_from_dict_roundtrip(self):
        """JobHandle serializes and deserializes correctly."""
        handle = JobHandle(
            job_id="job123",
            idempotency_key="key456",
            status="succeeded",
            output_path="/output/video.mp4",
            metadata={"cost_usd": 0.05},
        )
        
        d = handle.to_dict()
        restored = JobHandle.from_dict(d)
        
        assert restored.job_id == handle.job_id
        assert restored.idempotency_key == handle.idempotency_key
        assert restored.status == handle.status
        assert restored.output_path == handle.output_path
        assert restored.metadata == handle.metadata


# ─────────────────────────────────────────────
# InMemoryDedupeStore Tests
# ─────────────────────────────────────────────
class TestInMemoryDedupeStore:
    def test_put_and_get(self):
        """Store and retrieve job handle."""
        store = InMemoryDedupeStore()
        handle = JobHandle(job_id="job1", idempotency_key="key1")
        
        store.put(handle)
        retrieved = store.get("key1")
        
        assert retrieved is not None
        assert retrieved.job_id == "job1"

    def test_get_nonexistent_returns_none(self):
        """Get on nonexistent key returns None."""
        store = InMemoryDedupeStore()
        
        result = store.get("nonexistent")
        
        assert result is None

    def test_delete_existing(self):
        """Delete removes existing handle."""
        store = InMemoryDedupeStore()
        handle = JobHandle(job_id="job1", idempotency_key="key1")
        store.put(handle)
        
        deleted = store.delete("key1")
        
        assert deleted is True
        assert store.get("key1") is None

    def test_delete_nonexistent_returns_false(self):
        """Delete on nonexistent key returns False."""
        store = InMemoryDedupeStore()
        
        deleted = store.delete("nonexistent")
        
        assert deleted is False

    def test_clear(self):
        """Clear removes all handles."""
        store = InMemoryDedupeStore()
        store.put(JobHandle(job_id="j1", idempotency_key="k1"))
        store.put(JobHandle(job_id="j2", idempotency_key="k2"))
        store.put(JobHandle(job_id="j3", idempotency_key="k3"))
        
        count = store.clear()
        
        assert count == 3
        assert len(store) == 0

    def test_len_and_keys(self):
        """len() and keys() return correct values."""
        store = InMemoryDedupeStore()
        store.put(JobHandle(job_id="j1", idempotency_key="k1"))
        store.put(JobHandle(job_id="j2", idempotency_key="k2"))
        
        assert len(store) == 2
        assert set(store.keys()) == {"k1", "k2"}


class TestDedupeStoreGetOrCreate:
    def test_create_when_not_exists(self):
        """get_or_create creates new handle when key doesn't exist."""
        store = InMemoryDedupeStore()
        
        def create_fn():
            return ("new_job_id", "submitted")
        
        handle, created = store.get_or_create("new_key", create_fn)
        
        assert created is True
        assert handle.job_id == "new_job_id"
        assert handle.idempotency_key == "new_key"
        assert store.get("new_key") is not None

    def test_return_existing_when_active(self):
        """get_or_create returns existing handle when job is active."""
        store = InMemoryDedupeStore()
        existing = JobHandle(job_id="existing_job", idempotency_key="key1", status="running")
        store.put(existing)
        
        create_called = False
        def create_fn():
            nonlocal create_called
            create_called = True
            return ("new_job_id", "submitted")
        
        handle, created = store.get_or_create("key1", create_fn)
        
        assert created is False
        assert create_called is False
        assert handle.job_id == "existing_job"

    def test_return_existing_when_succeeded(self):
        """get_or_create returns existing handle when job succeeded (cache hit)."""
        store = InMemoryDedupeStore()
        existing = JobHandle(
            job_id="succeeded_job",
            idempotency_key="key1",
            status="succeeded",
            output_path="/output/video.mp4",
        )
        store.put(existing)
        
        create_called = False
        def create_fn():
            nonlocal create_called
            create_called = True
            return ("new_job_id", "submitted")
        
        handle, created = store.get_or_create("key1", create_fn)
        
        assert created is False
        assert create_called is False
        assert handle.job_id == "succeeded_job"
        assert handle.output_path == "/output/video.mp4"

    def test_create_new_when_failed(self):
        """get_or_create creates new handle when previous job failed (retry)."""
        store = InMemoryDedupeStore()
        failed = JobHandle(job_id="failed_job", idempotency_key="key1", status="failed")
        store.put(failed)
        
        def create_fn():
            return ("retry_job_id", "submitted")
        
        handle, created = store.get_or_create("key1", create_fn)
        
        assert created is True
        assert handle.job_id == "retry_job_id"


# ─────────────────────────────────────────────
# Duplicate Request Detection Tests
# ─────────────────────────────────────────────
class TestDuplicateRequestDetection:
    def test_duplicate_request_returns_existing_handle(self):
        """Duplicate request with same key returns existing job handle."""
        store = InMemoryDedupeStore()
        
        # First request creates job
        key = compute_idempotency_key("seg#1", "i2v", {"path": "/img.png"})
        handle1, created1 = store.get_or_create(key, lambda: ("job1", "submitted"))
        
        assert created1 is True
        
        # Mark job as running
        handle1.status = "running"
        store.put(handle1)
        
        # Duplicate request returns existing (no new job)
        handle2, created2 = store.get_or_create(key, lambda: ("job2", "submitted"))
        
        assert created2 is False
        assert handle2.job_id == "job1"  # Same job, not job2

    def test_changed_payload_creates_new_job(self):
        """Changed request payload produces new idempotency key and new job."""
        store = InMemoryDedupeStore()
        
        # First request
        key1 = compute_idempotency_key("seg#1", "i2v", {"path": "/img1.png"})
        handle1, created1 = store.get_or_create(key1, lambda: ("job1", "submitted"))
        handle1.status = "succeeded"
        store.put(handle1)
        
        # Changed payload = different key = new job
        key2 = compute_idempotency_key("seg#1", "i2v", {"path": "/img2.png"})
        handle2, created2 = store.get_or_create(key2, lambda: ("job2", "submitted"))
        
        assert key1 != key2
        assert created2 is True
        assert handle2.job_id == "job2"

    def test_changed_workflow_creates_new_job(self):
        """Changed workflow produces new idempotency key and new job."""
        store = InMemoryDedupeStore()
        payload = {"path": "/img.png"}
        
        # First request with i2v
        key1 = compute_idempotency_key("seg#1", "i2v", payload)
        handle1, created1 = store.get_or_create(key1, lambda: ("job1", "submitted"))
        handle1.status = "succeeded"
        store.put(handle1)
        
        # Same segment, different workflow = different key = new job
        key2 = compute_idempotency_key("seg#1", "digital_human", payload)
        handle2, created2 = store.get_or_create(key2, lambda: ("job2", "submitted"))
        
        assert key1 != key2
        assert created2 is True
        assert handle2.job_id == "job2"


# ─────────────────────────────────────────────
# Store Factory Tests
# ─────────────────────────────────────────────
class TestStoreFactory:
    def test_create_memory_store(self):
        """Factory creates InMemoryDedupeStore."""
        store = create_dedupe_store(store_type="memory")
        
        assert isinstance(store, InMemoryDedupeStore)

    def test_create_redis_store_without_url_raises(self):
        """Factory raises ValueError if redis requested without URL."""
        with pytest.raises(ValueError, match="redis_url is required"):
            create_dedupe_store(store_type="redis")

    def test_unknown_store_type_raises(self):
        """Factory raises ValueError for unknown store type."""
        with pytest.raises(ValueError, match="Unknown store_type"):
            create_dedupe_store(store_type="unknown")

    def test_get_default_store_returns_memory_without_redis_url(self, monkeypatch):
        """get_default_store returns InMemoryDedupeStore when PIXELLE_REDIS_URL not set."""
        monkeypatch.delenv("PIXELLE_REDIS_URL", raising=False)
        
        store = get_default_store()
        
        assert isinstance(store, InMemoryDedupeStore)


# ─────────────────────────────────────────────
# Integration with Existing Key Semantics
# ─────────────────────────────────────────────
class TestExistingKeySemantics:
    def test_segment_key_format_preserved(self):
        """Idempotency key uses existing segment_key format (content_key#occurrence)."""
        # Simulate segment_key format from models.py
        content_key = "abc123def456"
        occurrence_index = 2
        segment_key = f"{content_key}#{occurrence_index}"
        
        key = compute_idempotency_key(segment_key, "i2v", {"path": "/img.png"})
        
        assert len(key) == 32
        assert key == compute_idempotency_key(segment_key, "i2v", {"path": "/img.png"})

    def test_workflow_values_from_pixelle_capabilities(self):
        """Idempotency key works with all Pixelle workflow names."""
        workflows = ["digital_human", "i2v", "action_transfer"]
        payload = {"input": "/path/to/file"}
        
        keys = [
            compute_idempotency_key("seg#1", wf, payload)
            for wf in workflows
        ]
        
        # All keys should be unique
        assert len(set(keys)) == 3
        # All keys should be 32 chars
        assert all(len(k) == 32 for k in keys)
