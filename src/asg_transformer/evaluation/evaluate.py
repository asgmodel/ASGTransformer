from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from asg_transformer.config import settings
from asg_transformer.core.catalog import KnowledgeCatalog
from asg_transformer.models.semantic_encoder import SemanticEncoder

def evaluate(data_dir: Path, model_dir: Path, model_name: str, top_k: int=5) -> dict:
    catalog=KnowledgeCatalog(data_dir)
    encoder=SemanticEncoder(model_name,str(model_dir),settings.device,None)
    reciprocal=[]; recall=[]; lat=[]
    for item in catalog.techniques:
        start=time.perf_counter(); ranked=encoder.rank(item.description,catalog.techniques,"techniques",top_k); lat.append((time.perf_counter()-start)*1000)
        labels=[x.item.label for x in ranked]
        recall.append(float(item.label in labels)); reciprocal.append(1.0/(labels.index(item.label)+1) if item.label in labels else 0.0)
    return {"examples":len(catalog.techniques),f"recall@{top_k}":float(np.mean(recall)),"mrr":float(np.mean(reciprocal)),"latency_ms_avg":float(np.mean(lat)),"latency_ms_p95":float(np.percentile(lat,95))}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-dir",type=Path,default=settings.data_dir); p.add_argument("--model-dir",type=Path,default=settings.model_dir); p.add_argument("--model-name",default=settings.model_name); p.add_argument("--top-k",type=int,default=5); p.add_argument("--output",type=Path)
    a=p.parse_args(); result=evaluate(a.data_dir,a.model_dir,a.model_name,a.top_k); text=json.dumps(result,indent=2); print(text)
    if a.output: a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(text,encoding="utf-8")
if __name__=="__main__": main()
