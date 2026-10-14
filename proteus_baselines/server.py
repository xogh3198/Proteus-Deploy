"""Separate server process; no calls to an existing INFaaS/Proteus/Clipper service."""
import json
import logging
import os
import signal

from proteus_realtime.runtime import GpuRuntime
from proteus_realtime.server import create_server
from .serving import BaselineExecutor
from .config import verify_sources
from .sommelier import SommelierExecutor


def main():
    baseline = os.environ["BASELINE"]
    verify_sources(baseline=baseline)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [baseline] %(name)s %(message)s")
    common = dict(
        profile_path="/run/profiles/profiles.json", slo=float(os.environ.get("SLO_MS", "200")),
        batches=tuple(int(v) for v in os.environ.get("BATCH_SIZES", "1,2,4,8").split(",")),
        max_queue=int(os.environ.get("MAX_QUEUE_SIZE", "256")),
        interval=float(os.environ.get("ALLOCATION_INTERVAL", "10")))
    if baseline == "sommelier":
        executor = SommelierExecutor(GpuRuntime(), **common,
                                    guard_ms=float(os.environ.get("DEADLINE_GUARD_MS", "2")))
    else:
        executor = BaselineExecutor(GpuRuntime(), baseline, **common)
    server, _ = create_server(executor, "0.0.0.0:9000", int(os.environ.get("RPC_THREADS", "128")))
    server.start()
    logging.info("READY %s", json.dumps(executor.state(), ensure_ascii=False))
    def stop(*_):
        server.stop(5)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.wait_for_termination()
    finally:
        executor.close()


if __name__ == "__main__":
    main()
