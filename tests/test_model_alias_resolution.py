import unittest

from rotator_library.client._models import ModelsMixin


class _StaticProvider:
    def __init__(self, models):
        self._models = models

    def get_static_models(self):
        return list(self._models)


class _EmptyDefinitions:
    def get_all_provider_models(self, provider):
        return []


class _DummyClient(ModelsMixin):
    def __init__(self, providers):
        self.all_credentials = {provider: ["credential"] for provider in providers}
        self._providers = providers
        self._model_list_cache = {}
        self.model_definitions = _EmptyDefinitions()

    def _get_provider_instance(self, provider):
        models = self._providers.get(provider, [])
        return _StaticProvider(models)


class ModelAliasResolutionTests(unittest.TestCase):
    def test_resolves_unique_bare_model_alias_to_provider_model(self):
        client = _DummyClient({"zai": ["zai/glm-5.1", "zai/glm-5-turbo"]})

        self.assertEqual(client._resolve_model_alias("glm-5.1"), "zai/glm-5.1")

    def test_keeps_prefixed_model_unchanged(self):
        client = _DummyClient({"zai": ["zai/glm-5.1"]})

        self.assertEqual(client._resolve_model_alias(" zai/glm-5.1 "), "zai/glm-5.1")

    def test_leaves_ambiguous_bare_model_unresolved(self):
        client = _DummyClient(
            {
                "zai": ["zai/glm-5.1"],
                "other": ["other/glm-5.1"],
            }
        )

        with self.assertLogs("rotator_library", level="WARNING"):
            self.assertEqual(client._resolve_model_alias("glm-5.1"), "glm-5.1")


if __name__ == "__main__":
    unittest.main()
