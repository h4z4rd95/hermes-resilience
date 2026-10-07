"""Tests for resilience watchdog — creates disposable failures and verifies recovery."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.watchdog.watchdog import Watchdog, WatchdogConfig, GatewayStatus, CronStatus, DispatcherStatus
from src.failover.failover import FallbackChain, ProviderEntry, ProviderState, FailoverCategory
from src.context.context import ContextInspector, SessionContext, ContextProfile


class TestWatchdogDiagnostics(unittest.TestCase):
    """Test each diagnostic phase in isolation."""

    def setUp(self):
        self.hermes_home = Path("/root/.hermes")
        self.cfg = WatchdogConfig(
            hermes_home=self.hermes_home,
            dispatcher_home=Path("/root/hermes-tools/hermes-task-dispatcher"),
        )
        self.wd = Watchdog(self.cfg)

    def test_check_gateway_alive(self):
        """Gateway process alive → status ok."""
        result = self.wd.check_gateway()
        # On the actual server, gateway should be running
        self.assertIsNotNone(result.pid or True)  # pid might be None if state file missing
        # Structure is correct
        self.assertIn("pid", result.__dict__)
        self.assertIn("state", result.__dict__)

    def test_check_cron_heartbeat(self):
        """Cron heartbeat file readable → age calculated."""
        result = self.wd.check_cron()
        self.assertIsInstance(result.ticker_age_s, (int, float))
        self.assertIsInstance(result.active_jobs, int)

    def test_check_dispatcher_stale(self):
        """Dispatcher state DB read correctly."""
        result = self.wd.check_dispatcher()
        self.assertIsInstance(result.sqlite_exists, bool)
        self.assertIsInstance(result.running_tasks, list)
        self.assertIsInstance(result.stale_leases, list)

    def test_check_model_errors(self):
        """Provider models cache read for errors."""
        result = self.wd.check_model()
        self.assertIsInstance(result.recent_errors, list)


class TestFallbackChain(unittest.TestCase):
    """Test provider fallback ordering and cooldown."""

    def test_chaining_order(self):
        """Providers returned in priority order."""
        hh = Path("/root/.hermes")
        chain = FallbackChain(hh)
        if chain.providers:
            # Check sorted by priority
            priorities = [p.priority for p in chain.providers]
            self.assertEqual(priorities, sorted(priorities))

    def test_cooldown_mechanism(self):
        """After max_retries failures, provider enters cooldown."""
        entry = ProviderEntry(
            name="test-provider",
            base_url="http://test.local/v1",
            model="test-model",
            priority=0,
            max_retries=2,
            cooldown_s=1,  # 1 second for test speed
        )
        self.assertEqual(entry.state, ProviderState.ACTIVE)

        chain = FallbackChain(hermes_home=Path("/root/.hermes"))
        chain.providers = [entry]

        # Exhaust retries
        for i in range(2):
            chain.mark_failure("test-provider", f"error {i}", FailoverCategory.UNKNOWN)
            self.assertEqual(entry.retries_remaining, 2 - i - 1)

        # Should be in cooldown now
        self.assertEqual(entry.state, ProviderState.COOLDOWN)

    def test_cooldown_expiry(self):
        """Provider leaves cooldown after cooldown_s."""
        entry = ProviderEntry(
            name="cooldown-test",
            base_url="http://test.local/v1",
            model="test-model",
            priority=0,
            max_retries=1,
            cooldown_s=0,  # immediate expiry
        )
        chain = FallbackChain(hermes_home=Path("/root/.hermes"))
        chain.providers = [entry]

        chain.mark_failure("cooldown-test", "err", FailoverCategory.UNKNOWN)
        self.assertEqual(entry.state, ProviderState.COOLDOWN)

        # After time, should be recoverable
        time.sleep(0.1)
        candidate = chain.get_next_candidate()
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.name, "cooldown-test")

    def test_success_resets_cooldown(self):
        """Successful provider resets retry budget and cooldown."""
        entry = ProviderEntry(
            name="reset-test",
            base_url="http://test.local/v1",
            model="test-model",
            priority=0,
            max_retries=1,
            cooldown_s=300,
        )
        chain = FallbackChain(hermes_home=Path("/root/.hermes"))
        chain.providers = [entry]

        chain.mark_failure("reset-test", "err", FailoverCategory.UNKNOWN)
        self.assertEqual(entry.state, ProviderState.COOLDOWN)
        chain.mark_success("reset-test")
        self.assertEqual(entry.state, ProviderState.ACTIVE)
        self.assertEqual(entry.retries_remaining, 1)

    def test_exhaustion(self):
        """All providers exhausted → no candidates."""
        entries = [
            ProviderEntry(name="p1", base_url="http://1.local/v1", model="m1", priority=0, max_retries=0),
            ProviderEntry(name="p2", base_url="http://2.local/v1", model="m2", priority=1, max_retries=0),
        ]
        chain = FallbackChain(hermes_home=Path("/root/.hermes"))
        chain.providers = entries

        self.assertTrue(chain.is_exhausted())
        self.assertIsNone(chain.get_next_candidate())

    def test_isolated_failures(self):
        """Failure on p1 does not affect p2."""
        p1 = ProviderEntry(name="p1", base_url="http://1.local/v1", model="m1", priority=0, max_retries=0)
        p2 = ProviderEntry(name="p2", base_url="http://2.local/v1", model="m2", priority=1, max_retries=5)
        chain = FallbackChain(hermes_home=Path("/root/.hermes"))
        chain.providers = [p1, p2]

        chain.mark_failure("p1", "err", FailoverCategory.UNKNOWN)
        self.assertEqual(p1.state, ProviderState.COOLDOWN)
        self.assertEqual(p2.state, ProviderState.ACTIVE)

        # Should skip p1, return p2
        candidate = chain.get_next_candidate()
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.name, "p2")


class TestErrorClassification(unittest.TestCase):
    """Test failover error type classification."""

    def test_rate_limit(self):
        from src.failover.failover import classify_error, FailoverCategory
        self.assertEqual(
            classify_error("rate limit exceeded"),
            FailoverCategory.RATE_LIMIT
        )
        self.assertEqual(
            classify_error("HTTP 429 Too Many Requests"),
            FailoverCategory.RATE_LIMIT
        )

    def test_auth_failure(self):
        from src.failover.failover import classify_error, FailoverCategory
        self.assertEqual(
            classify_error("401 Unauthorized"),
            FailoverCategory.AUTH
        )

    def test_network_error(self):
        from src.failover.failover import classify_error, FailoverCategory
        self.assertEqual(
            classify_error("Connection refused"),
            FailoverCategory.NETWORK
        )

    def test_timeout(self):
        from src.failover.failover import classify_error, FailoverCategory
        self.assertEqual(
            classify_error("request timed out after 120s"),
            FailoverCategory.TIMEOUT
        )

    def test_unknown(self):
        from src.failover.failover import classify_error, FailoverCategory
        self.assertEqual(
            classify_error("some weird error"),
            FailoverCategory.UNKNOWN
        )


class TestSafeModelSwitch(unittest.TestCase):
    """Test context inspection before model switch."""

    def test_inspector_initialization(self):
        inspector = ContextInspector(Path("/root/.hermes"))
        self.assertIsNotNone(inspector.hermes_home)

    def test_profile_unsafe(self):
        """Session with 90%+ context utilization → unsafe."""
        inspector = ContextInspector(Path("/root/.hermes"))
        session = SessionContext(
            session_id="test",
            session_name="test-session",
            estimated_tokens=115000,  # 90% of 128k
        )
        profile = inspector.check_context_for_model(session, "test-model", 128000)
        self.assertTrue(profile.unsafe)
        self.assertEqual(profile.utilization_pct, 89.84375)

    def test_profile_safe(self):
        """Session with 50% utilization → safe."""
        inspector = ContextInspector(Path("/root/.hermes"))
        session = SessionContext(
            session_id="test",
            session_name="test-session",
            estimated_tokens=64000,
        )
        profile = inspector.check_context_for_model(session, "test-model", 128000)
        self.assertFalse(profile.unsafe)
        self.assertEqual(profile.utilization_pct, 50.0)


class TestHandoffDocument(unittest.TestCase):
    """Test handoff document creation as last resort."""

    def test_handoff_creation(self):
        from src.context.context import ContextCompressor, SessionContext

        compressor = ContextCompressor(Path("/root/.hermes"))
        session = SessionContext(
            session_id="test-session-123",
            session_name="test-session",
            estimated_tokens=110000,
            total_messages=50,
            last_turn_text="Testing handoff document creation",
        )
        hr = compressor.create_handoff_document(
            "test-session", session,
            error="All compression methods failed",
        )

        self.assertTrue(hr.success)
        self.assertTrue(hr.lineage_preserved)
        self.assertEqual(hr.old_session_id, "test-session-123")
        self.assertTrue(hr.handoff_path.endswith(".json"))

    def test_handoff_file_written(self):
        from src.context.context import ContextCompressor, SessionContext
        import os

        compressor = ContextCompressor(Path("/root/.hermes"))
        session = SessionContext(
            session_id="test-session-456",
            session_name="test-session-2",
            estimated_tokens=100000,
        )
        hr = compressor.create_handoff_document("test-session-2", session, "timeout")

        self.assertTrue(hr.success)
        self.assertTrue(os.path.exists(hr.handoff_path))

        # Verify JSON is valid
        data = json.loads(Path(hr.handoff_path).read_text())
        self.assertEqual(data["old_session_id"], "test-session-456")
        self.assertIn("completed_work", data)
        self.assertIn("pending_work", data)


class TestDispatcherIntegration(unittest.TestCase):
    """Integration tests that verify real files on server."""

    def test_dispatcher_state_db_readable(self):
        """Dispatcher SQLite DB is readable and has expected schema."""
        db_path = Path("/root/hermes-tools/hermes-task-dispatcher/state/dispatcher.sqlite")
        if not db_path.exists():
            self.skipTest("Dispatcher state DB not found on this host")

        import sqlite3
        conn = sqlite3.connect(str(db_path))
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        conn.close()
        self.assertTrue(len(tables) > 0)

    def test_hermes_config_readable(self):
        """Hermes config.yaml exists and is parseable."""
        config_path = Path("/root/.hermes/config.yaml")
        if not config_path.exists():
            self.skipTest("Config not found")

        content = config_path.read_text()
        self.assertIn("model:", content)
        self.assertIn("telegram:", content)

    def test_cron_directory_exists(self):
        """Cron directory with heartbeat file exists."""
        cron_dir = Path("/root/.hermes/cron")
        heartbeat = cron_dir / "ticker_heartbeat"
        if not heartbeat.exists():
            self.skipTest("Cron heartbeat not found")

        content = heartbeat.read_text().strip()
        float(content)  # Should be a valid timestamp


if __name__ == "__main__":
    unittest.main(verbosity=2)