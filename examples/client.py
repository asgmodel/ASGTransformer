import requests
payload={"text":"Remote access followed by manipulation of an industrial controller","max_steps":6}
r=requests.post("http://localhost:8000/v1/scenarios/generate",json=payload,timeout=60); r.raise_for_status(); print(r.json())
