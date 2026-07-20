from __future__ import annotations
import argparse, json
from dataclasses import asdict
from asg_transformer import __version__
from asg_transformer.config import settings
from asg_transformer.core.service import get_service

def main() -> None:
    p=argparse.ArgumentParser(prog="asg",description="ASG Transformer command line interface"); p.add_argument("--version",action="version",version=__version__)
    sub=p.add_subparsers(dest="command",required=True)
    serve=sub.add_parser("serve"); serve.add_argument("--host",default="0.0.0.0"); serve.add_argument("--port",type=int,default=8000); serve.add_argument("--reload",action="store_true")
    pred=sub.add_parser("predict"); pred.add_argument("text"); pred.add_argument("--task",choices=["technique","software","group"],default="technique"); pred.add_argument("--top-k",type=int,default=5)
    scen=sub.add_parser("scenario"); scen.add_argument("text"); scen.add_argument("--max-steps",type=int,default=8)
    sub.add_parser("doctor")
    args,extra=p.parse_known_args()
    if args.command=="serve":
        import uvicorn; uvicorn.run("asg_transformer.api.main:app",host=args.host,port=args.port,reload=args.reload)
    elif args.command=="predict":
        print(json.dumps([{"label":x.item.label,"score":x.score,"description":x.item.description} for x in get_service().classify(args.text,args.task,args.top_k)],indent=2))
    elif args.command=="scenario":
        steps=get_service().generator.generate(args.text,args.max_steps,5,.35); print(json.dumps([asdict(s) for s in steps],indent=2))
    elif args.command=="doctor":
        checks={"version":__version__,"data_dir":str(settings.data_dir),"data_exists":settings.data_dir.exists(),"model_dir":str(settings.model_dir),"trained_model_exists":settings.model_dir.exists(),"device":settings.device}; print(json.dumps(checks,indent=2)); raise SystemExit(0 if checks["data_exists"] else 1)
if __name__=="__main__": main()
