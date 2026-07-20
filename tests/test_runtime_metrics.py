from asg_transformer.observability.metrics import RuntimeMetrics

def test_runtime_metrics_snapshot():
    m=RuntimeMetrics(); m.record('/health',10.0,200); m.record('/x',30.0,500); s=m.snapshot(); assert s['requests']==2; assert s['server_errors']==1; assert s['latency_ms']['avg']==20.0
