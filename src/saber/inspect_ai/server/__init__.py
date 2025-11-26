"""Server lifecycle and health management components."""

from .domain_manager import get_active_domain, register_domain, remove_active_domain, unregister_domain_on_failure
from .health_check import wait_for_server_health
from .preflight import run_preflight_check
from .server import DomainContext, DomainController

__all__ = [
    "DomainController",
    "DomainContext",
    "get_active_domain",
    "register_domain",
    "remove_active_domain",
    "unregister_domain_on_failure",
    "wait_for_server_health",
    "run_preflight_check",
]
