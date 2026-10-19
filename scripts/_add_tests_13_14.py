"""One-shot script: inserts test_13 and test_14 into test_readiness_gate_nemotron.py."""
import pathlib

path = pathlib.Path("tests/test_readiness_gate_nemotron.py")

new_tests = '''\
    # 13. No-key case fails closed even when the developer shell has a key set
    def test_13_no_key_fails_closed_even_when_shell_has_key(self):
        """Provider reports not configured when NEBIUS_API_KEY is cleared, regardless of shell state.

        Deterministic environment-isolation regression: even if the developer shell exports
        NEBIUS_API_KEY (e.g. after a successful live smoke test), clearing it within the test
        scope must make NebiusNemotronProvider(client=None).is_configured() return False and
        generate() raise RuntimeError.  Zero network calls are made.
        """
        with patch.dict(os.environ, {"NEBIUS_API_KEY": ""}, clear=False):
            provider = NebiusNemotronProvider(client=None)
            self.assertFalse(
                provider.is_configured(),
                "Provider must report unconfigured when NEBIUS_API_KEY is empty, "
                "even if the outer shell has the key set.",
            )
            with self.assertRaises(RuntimeError):
                provider.generate(prompt="Should fail closed.")

    # 14. Valid configuration enables the provider without network calls
    def test_14_valid_configuration_enables_provider_without_network_calls(self):
        """Provider reports configured when a client is injected; env key controls the client=None path.

        Two sub-cases:
        (a) Injected mock client -- is_configured() is True regardless of env.
        (b) client=None with a sentinel env key -- is_configured() is True, key must not leak.
        No live network calls are made.
        """
        # (a) Injected mock client -- always configured
        mock_client = MagicMock()
        provider_a = NebiusNemotronProvider(client=mock_client)
        self.assertTrue(provider_a.is_configured())

        # (b) client=None path with a clearly fake sentinel key
        sentinel_key = "sk-test-DETERMINISTIC-SENTINEL-NOT-REAL-0000"
        with patch.dict(os.environ, {"NEBIUS_API_KEY": sentinel_key}, clear=False):
            provider_b = NebiusNemotronProvider(client=None)
            self.assertTrue(
                provider_b.is_configured(),
                "Provider must report configured when NEBIUS_API_KEY is non-empty.",
            )
            # Sentinel key must not leak into repr or model_id
            self.assertNotIn(sentinel_key, repr(provider_b))
            self.assertNotIn(sentinel_key, provider_b.model_id)

'''

content = path.read_text(encoding="utf-8")
marker = '\nif __name__ == "__main__":'
idx = content.find(marker)
assert idx != -1, f"Marker not found in {path}"
new_content = content[:idx] + "\n" + new_tests + content[idx:]
path.write_text(new_content, encoding="utf-8")
total_lines = new_content.count("\n") + 1
print(f"Done. Total lines: {total_lines}")
