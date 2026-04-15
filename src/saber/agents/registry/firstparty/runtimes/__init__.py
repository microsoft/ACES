"""First-party runtime definitions.

Importing this package registers all built-in runtimes with the RuntimeRegistry.
"""

from saber.agents.registry.firstparty.runtime_registry import RuntimeRegistry
from saber.agents.registry.firstparty.runtimes.hyenas import HYENAS_RUNTIME

if not RuntimeRegistry.get(HYENAS_RUNTIME.name):
    RuntimeRegistry.register(HYENAS_RUNTIME)
