"""Read-only metric decomposition. Never used for admission, SLO timers or routing."""
from statistics import fmean


def add_boundaries(row):
    if not row.get("ok"):return row
    t=row["timings"]
    if "worker.cpu_service" in t:
        prep=float(t["worker.cpu_service"])
    elif "worker.decode" in t and "worker.preprocess" in t:
        prep=float(t["worker.decode"])+float(t["worker.preprocess"])
    else:return row
    t["measurement.preprocessing_wall"]=prep
    derived=float(row["e2e_ms"])-prep
    if derived>=0:t["measurement.e2e_minus_preprocessing"]=derived
    return row


def boundary_report(rows):
    def quantile(values,p):
        values=sorted(values)
        if not values:return None
        i=(len(values)-1)*p
        a=int(i);b=min(a+1,len(values)-1)
        return values[a]+(values[b]-values[a])*(i-a)
    report={}
    for name,key in (("native_api_latency","measurement.native_api_latency"),
        ("worker_callback_latency","measurement.worker_callback_latency"),
        ("preprocessing_wall","measurement.preprocessing_wall"),
        ("after_preprocessing_tail","measurement.post_preprocess"),
        ("derived_e2e_minus_preprocessing","measurement.e2e_minus_preprocessing")):
        values=[float(r["timings"][key]) for r in rows if r.get("ok") and key in r.get("timings",{})]
        report[name]={"samples":len(values),"mean_ms":fmean(values) if values else None,
            "p50_ms":quantile(values,.5),"p95_ms":quantile(values,.95),"p99_ms":quantile(values,.99)}
    report["definitions"]={
        "native_api_latency":"successful native response latency field, when available; its source API boundary is preserved",
        "worker_callback_latency":"Clipper model callback entry -> worker result, excluding earlier native frontend/batch queue",
        "preprocessing_wall":"native decode/preprocessing interval; DALI may run part on GPU",
        "after_preprocessing_tail":"recorded preprocessing completion -> worker result; queues/load occurring earlier are excluded",
        "derived_e2e_minus_preprocessing":"arithmetic decomposition of the same run, NOT a run without preprocessing and NOT a replacement SLO",
        "slo":"actual client E2E success/failure accounting remains unchanged"}
    return report
