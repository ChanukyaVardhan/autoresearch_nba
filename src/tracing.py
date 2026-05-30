"""Tracing layer for observability (Raindrop Workshop / any OTLP backend).

Emits OpenTelemetry spans for each autoresearch iteration so the whole loop is
visible in a live debugger UI: the Codex prompt, its proposal, the leakage check,
training, validation metrics, diagnostics, and the keep/revert verdict.

Raindrop Workshop (`raindrop workshop`, localhost:5899) consumes OTLP traces, so
once it's running this "just works". Setup at the workshop:
    curl -fsSL https://raindrop.sh/install | bash
    raindrop workshop                 # starts the local UI + OTLP collector
    # then run the loop with the endpoint pointed at Workshop (auto-detected below)

Degrades gracefully: if OpenTelemetry isn't installed OR no endpoint is configured,
every call is a cheap no-op and the loop runs unchanged. Nothing here can break the
experiment.

Endpoint precedence (first set wins):
  RAINDROP_LOCAL_DEBUGGER  (Raindrop's own var) > OTEL_EXPORTER_OTLP_ENDPOINT
  > http://localhost:5899  (Raindrop Workshop default, used if --trace is forced on)
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any, Optional

_SERVICE = "autoresearch-nba"
_DEFAULT_ENDPOINT = "http://localhost:5899"


def _resolve_endpoint(force: bool) -> Optional[str]:
    ep = os.environ.get("RAINDROP_LOCAL_DEBUGGER") or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if ep:
        return ep
    return _DEFAULT_ENDPOINT if force else None


def _traces_url(endpoint: str) -> str:
    """Build the OTLP /v1/traces URL from a base that may already include /v1 or /v1/.
    Raindrop gives RAINDROP_LOCAL_DEBUGGER=http://host:5899/v1/ -> .../v1/traces.
    A bare host (http://host:5899) -> http://host:5899/v1/traces."""
    e = endpoint.rstrip("/")
    if e.endswith("/v1"):
        return e + "/traces"
    if e.endswith("/v1/traces"):
        return e
    return e + "/v1/traces"


def _flatten(prefix: str, obj: Any, out: dict) -> None:
    """Flatten nested dicts into dotted span-attribute keys (OTel attrs are scalar)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(obj, (list, tuple)):
        out[prefix] = json.dumps(obj)[:4000]
    elif isinstance(obj, (str, bool, int, float)) or obj is None:
        out[prefix] = obj if obj is not None else ""
    else:
        out[prefix] = str(obj)[:4000]


class _NoopSpan:
    def set(self, **kw): return self
    def set_attrs(self, d, prefix=""): return self
    def __enter__(self): return self
    def __exit__(self, *a): return False


class TraceLogger:
    """Thin OTel wrapper. One TraceLogger per autoresearch run."""

    def __init__(self, run_name: str, enabled: bool = True, force_endpoint: bool = False):
        self._tracer = None
        self.endpoint = None
        if not enabled:
            return
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        except ImportError:
            return  # OTel not installed -> no-op
        endpoint = _resolve_endpoint(force_endpoint)
        if endpoint is None:
            return  # nothing to export to -> no-op (no UI running)
        self.endpoint = endpoint
        provider = TracerProvider(resource=Resource.create({
            "service.name": _SERVICE, "run.name": run_name,
        }))
        provider.add_span_processor(BatchSpanProcessor(
            OTLPSpanExporter(endpoint=_traces_url(endpoint))))
        # use a private provider so we don't stomp a globally-set one
        self._tracer = provider.get_tracer(_SERVICE)
        self._provider = provider

    @property
    def active(self) -> bool:
        return self._tracer is not None

    @contextmanager
    def span(self, name: str, attrs: Optional[dict] = None):
        if self._tracer is None:
            yield _NoopSpan(); return
        with self._tracer.start_as_current_span(name) as sp:
            wrapper = _SpanWrapper(sp)
            if attrs:
                wrapper.set_attrs(attrs)
            yield wrapper

    def shutdown(self) -> None:
        if self._tracer is not None:
            self._provider.shutdown()


class _SpanWrapper:
    def __init__(self, sp):
        self._sp = sp

    def set(self, **kw):
        for k, v in kw.items():
            self.set_attrs({k: v})
        return self

    def set_attrs(self, d: dict, prefix: str = ""):
        flat: dict = {}
        _flatten(prefix, d, flat)
        for k, v in flat.items():
            try:
                self._sp.set_attribute(k, v)
            except Exception:
                self._sp.set_attribute(k, str(v)[:4000])
        return self
