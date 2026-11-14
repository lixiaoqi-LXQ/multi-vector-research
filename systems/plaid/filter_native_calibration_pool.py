"""Retain successful finite calibration records for native latency selection; no holdout reads."""
import argparse,hashlib,json,math
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();source=Path(a.source);out=Path(a.output);rows=json.loads(source.read_text());kept=[];excluded=[]
 for row in rows:
  valid=not row.get('failed') and all(isinstance(row.get(k),(int,float)) and math.isfinite(row[k]) for k in ['mean_ms','R10','R100'])
  (kept if valid else excluded).append(row)
 assert kept,'No successful finite calibration records'
 payload=json.dumps(kept,indent=2)
 out.parent.mkdir(parents=True,exist_ok=True)
 if out.exists():assert out.read_text()==payload,'Refusing to change prior calibration pool'
 else:out.write_text(payload)
 provenance={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'kept':len(kept),'excluded_failed_calibration_records':excluded,'rule':'only successful finite calibration rows retained; no recall-threshold or holdout-based filtering','heldout_read':False}
 out.with_suffix('.provenance.json').write_text(json.dumps(provenance,indent=2))
if __name__=='__main__':main()
