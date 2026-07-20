from __future__ import annotations
import time
from collections import defaultdict
from threading import Lock

class RuntimeMetrics:
    def __init__(self) -> None:
        self._lock = Lock(); self._requests=0; self._errors=0; self._latencies=[]; self._routes=defaultdict(int)
    def record(self, route: str, latency_ms: float, status_code: int) -> None:
        with self._lock:
            self._requests += 1; self._routes[route] += 1; self._latencies.append(latency_ms)
            if status_code >= 500: self._errors += 1
            if len(self._latencies)>5000: self._latencies=self._latencies[-2500:]
    def snapshot(self) -> dict:
        with self._lock:
            values=sorted(self._latencies)
            def pct(p): return values[min(len(values)-1,int((len(values)-1)*p))] if values else 0.0
            return {"requests":self._requests,"server_errors":self._errors,"latency_ms":{"avg":sum(values)/len(values) if values else 0.0,"p50":pct(.5),"p95":pct(.95),"p99":pct(.99)},"routes":dict(self._routes)}
metrics=RuntimeMetrics()
