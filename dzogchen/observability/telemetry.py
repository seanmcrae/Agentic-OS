"""Telemetry: distributed tracing, metrics collection, and structured event logging."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Event:
    name: str
    timestamp: float
    attributes: Dict[str, Any]
    level: str = "INFO"  # "INFO", "WARN", "ERROR"


@dataclass
class Metric:
    name: str
    metric_type: MetricType
    value: float
    unit: str
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)


class Span:
    """A tracing span that works as both a sync and async context manager."""

    def __init__(
        self,
        name: str,
        telemetry: "Telemetry",
        parent_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.telemetry = telemetry
        self.parent_id = parent_id
        self.span_id: str = uuid.uuid4().hex
        self.trace_id: str = uuid.uuid4().hex
        self.start_time: float = time.time()
        self.end_time: Optional[float] = None
        self.status: str = "OK"
        self.status_message: str = ""
        self.events: List[Dict[str, Any]] = []
        self.attributes: Dict[str, Any] = dict(attributes) if attributes else {}

    # ---- sync context manager ------------------------------------------------

    def __enter__(self) -> "Span":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.end_time = time.time()
        if exc_type is not None:
            self.set_status("ERROR", str(exc_val))
        self.telemetry._record_span(self)
        return False

    # ---- async context manager -----------------------------------------------

    async def __aenter__(self) -> "Span":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.end_time = time.time()
        if exc_type is not None:
            self.set_status("ERROR", str(exc_val))
        self.telemetry._record_span(self)
        return False

    # ---- span API ------------------------------------------------------------

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_status(self, status: str, message: str = "") -> None:
        self.status = status
        self.status_message = message

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        self.events.append(
            {
                "name": name,
                "timestamp": time.time(),
                "attributes": attributes or {},
            }
        )

    @property
    def elapsed_ms(self) -> float:
        end = self.end_time if self.end_time is not None else time.time()
        return (end - self.start_time) * 1000.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "elapsed_ms": self.elapsed_ms,
            "status": self.status,
            "status_message": self.status_message,
            "attributes": self.attributes,
            "events": self.events,
        }


class Telemetry:
    """Thread-safe telemetry collector for spans, metrics, and events."""

    def __init__(
        self,
        service_name: str = "dzogchen",
        export_to_file: Optional[str] = None,
        max_spans: int = 10000,
    ) -> None:
        self.service_name = service_name
        self.export_to_file = export_to_file
        self.max_spans = max_spans

        self._lock = threading.Lock()
        self._spans: List[Span] = []
        self._metrics: List[Metric] = []
        self._events: List[Event] = []

    # ---- span creation -------------------------------------------------------

    def span(self, name: str, **attributes: Any) -> Span:
        """Create a new span. Use as a context manager."""
        return Span(name=name, telemetry=self, attributes=attributes if attributes else None)

    def _record_span(self, span: Span) -> None:
        with self._lock:
            self._spans.append(span)
            if len(self._spans) > self.max_spans:
                self._spans = self._spans[-self.max_spans:]

        if self.export_to_file:
            self._append_to_file({"type": "span", "data": span.to_dict()})

    # ---- metrics -------------------------------------------------------------

    def counter(self, name: str, value: float = 1, labels: Optional[Dict[str, str]] = None) -> None:
        self._record_metric(name, MetricType.COUNTER, value, "", labels)

    def gauge(self, name: str, value: float, unit: str = "", labels: Optional[Dict[str, str]] = None) -> None:
        self._record_metric(name, MetricType.GAUGE, value, unit, labels)

    def histogram(self, name: str, value: float, unit: str = "", labels: Optional[Dict[str, str]] = None) -> None:
        self._record_metric(name, MetricType.HISTOGRAM, value, unit, labels)

    def timer(self, name: str, value_ms: float, labels: Optional[Dict[str, str]] = None) -> None:
        self._record_metric(name, MetricType.TIMER, value_ms, "ms", labels)

    def _record_metric(
        self,
        name: str,
        metric_type: MetricType,
        value: float,
        unit: str,
        labels: Optional[Dict[str, str]],
    ) -> None:
        metric = Metric(
            name=name,
            metric_type=metric_type,
            value=value,
            unit=unit,
            timestamp=time.time(),
            labels=labels or {},
        )
        with self._lock:
            self._metrics.append(metric)

        if self.export_to_file:
            self._append_to_file(
                {
                    "type": "metric",
                    "data": {
                        "name": metric.name,
                        "metric_type": metric.metric_type.value,
                        "value": metric.value,
                        "unit": metric.unit,
                        "timestamp": metric.timestamp,
                        "labels": metric.labels,
                    },
                }
            )

    # ---- events --------------------------------------------------------------

    def log_event(self, name: str, level: str = "INFO", **attributes: Any) -> None:
        event = Event(
            name=name,
            timestamp=time.time(),
            attributes=attributes,
            level=level,
        )
        with self._lock:
            self._events.append(event)

        if self.export_to_file:
            self._append_to_file(
                {
                    "type": "event",
                    "data": {
                        "name": event.name,
                        "timestamp": event.timestamp,
                        "attributes": event.attributes,
                        "level": event.level,
                    },
                }
            )

    # ---- querying ------------------------------------------------------------

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Aggregate metrics: sum counters, last gauge value, p95/max/avg for histograms and timers."""
        with self._lock:
            metrics_snapshot = list(self._metrics)

        counters: Dict[str, float] = defaultdict(float)
        gauges: Dict[str, float] = {}
        distributions: Dict[str, List[float]] = defaultdict(list)

        for m in metrics_snapshot:
            if m.metric_type == MetricType.COUNTER:
                counters[m.name] += m.value
            elif m.metric_type == MetricType.GAUGE:
                gauges[m.name] = m.value  # last value wins
            elif m.metric_type in (MetricType.HISTOGRAM, MetricType.TIMER):
                distributions[m.name].append(m.value)

        summary: Dict[str, Any] = {
            "counters": dict(counters),
            "gauges": dict(gauges),
            "distributions": {},
        }

        for name, values in distributions.items():
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            avg = sum(sorted_vals) / n if n else 0.0
            p95_idx = int(n * 0.95) if n > 1 else n - 1
            summary["distributions"][name] = {
                "count": n,
                "avg": avg,
                "p95": sorted_vals[p95_idx] if sorted_vals else 0.0,
                "max": sorted_vals[-1] if sorted_vals else 0.0,
                "min": sorted_vals[0] if sorted_vals else 0.0,
            }

        return summary

    def get_recent_spans(self, n: int = 50) -> List[dict]:
        with self._lock:
            recent = self._spans[-n:]
        return [s.to_dict() for s in recent]

    def get_recent_events(self, n: int = 100) -> List[dict]:
        with self._lock:
            recent = self._events[-n:]
        return [
            {
                "name": e.name,
                "timestamp": e.timestamp,
                "attributes": e.attributes,
                "level": e.level,
            }
            for e in recent
        ]

    async def export(self) -> dict:
        """Export all telemetry as a JSON-serializable dict."""
        with self._lock:
            spans_data = [s.to_dict() for s in self._spans]
            metrics_data = [
                {
                    "name": m.name,
                    "metric_type": m.metric_type.value,
                    "value": m.value,
                    "unit": m.unit,
                    "timestamp": m.timestamp,
                    "labels": m.labels,
                }
                for m in self._metrics
            ]
            events_data = [
                {
                    "name": e.name,
                    "timestamp": e.timestamp,
                    "attributes": e.attributes,
                    "level": e.level,
                }
                for e in self._events
            ]

        payload = {
            "service_name": self.service_name,
            "exported_at": time.time(),
            "spans": spans_data,
            "metrics": metrics_data,
            "events": events_data,
            "summary": self.get_metrics_summary(),
        }

        if self.export_to_file:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._write_export_file, payload)
            except Exception:
                pass

        return payload

    # ---- internal helpers ----------------------------------------------------

    def _append_to_file(self, record: dict) -> None:
        if not self.export_to_file:
            return
        try:
            with self._lock:
                with open(self.export_to_file, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record) + "\n")
        except Exception:
            pass

    def _write_export_file(self, payload: dict) -> None:
        if not self.export_to_file:
            return
        export_path = self.export_to_file
        if not export_path.endswith(".json"):
            export_path = export_path + ".full.json"
        try:
            with open(export_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        except Exception:
            pass
