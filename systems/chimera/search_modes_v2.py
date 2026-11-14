"""Explicit executable modes and calibration-family identity, CPU only."""
import json
KERNEL_MODES=('full','no-lut','no-aggregation','neither')
ALGORITHM_MODES=('full','no-cr','no-ivf-pg','neither')
def modes(spec):
 k=spec.get('kernel_mode','full');a=spec.get('algorithm_variant','full');c=spec.get('expected_coarse_clusters')
 if k not in KERNEL_MODES or a not in ALGORITHM_MODES:raise ValueError('Unknown search mode')
 flat=a in ('no-ivf-pg','neither')
 if flat:
  if type(c) is not int or c<=0:raise ValueError('Flat routing requires positive explicit expected_coarse_clusters')
 elif c is not None:raise ValueError('Original CAGRA route forbids expected_coarse_clusters')
 return dict(kernel_mode=k,algorithm_variant=a,expected_coarse_clusters=c)
def selection_family(meta):
 s=meta['spec'];m=modes(s)
 if meta.get('search_modes',m)!=m:raise ValueError('Prepared/spec mode mismatch')
 return dict(search_modes=m,binary_sha256=meta['binary_sha256'],index_identity=meta['index_identity'],k=meta['k'],query_payload_sha256=meta['query_payload_sha256'],gt_ids_sha256=meta['gt_ids_sha256'],GT_manifest=meta['GT_manifest'],GT_manifest_sha256=meta.get('GT_manifest_sha256'),calibration_manifest_sha256=meta.get('query_rows_manifest_sha256'),calibration_rows=s.get('query_rows'),split_manifest=s.get('split_manifest'),CPU_affinity=meta['CPU_affinity'],OpenMP_environment=meta['OpenMP_environment'],load_mode=meta['load_mode'],warmup=s.get('warmup',5))

def verify_mode_log(path,spec):
 import re
 text=__import__('pathlib').Path(path).read_text()
 rows=re.findall(r'^\[VARIANT\] algorithm_variant=(\S+) kernel_mode=(\S+) use_lut=([01]) use_aggregation=([01])$',text,re.M)
 declared=modes(spec);explicit='kernel_mode' in spec or 'algorithm_variant' in spec
 if not rows and not explicit:return dict(verified_legacy_full=True,**declared)
 if len(rows)!=1:raise ValueError('Require unique actual VARIANT mode receipt')
 a,k,lut,agg=rows[0]
 if a!=declared['algorithm_variant'] or k!=declared['kernel_mode']:raise ValueError('Actual engine mode differs from spec')
 if int(lut)!=int(k not in ('no-lut','neither')) or int(agg)!=int(k not in ('no-aggregation','neither')):raise ValueError('Mode/LUT/aggregation receipt mismatch')
 return dict(verified=True,**declared,use_lut=int(lut),use_aggregation=int(agg))

def validate_points(spec,nprobe_limit,docs=None):
 """CLI positivity/capacity contract; no-CR does not consume k_refine."""
 cr=modes(spec)['algorithm_variant'] not in ('no-cr','neither');result=[]
 for p in spec['points']:
  for key in ('nprobe','k_refine','k_full_bit','num_chunks'):
   if type(p[key]) is not int or p[key]<=0:raise ValueError('Positive integer search parameter required: '+key)
  if p['nprobe']>nprobe_limit:raise ValueError('nprobe exceeds kernel capacity')
  itopk=p.get('cagra_itopk_size',min(p['nprobe']+32,512))
  if type(itopk) is not int or itopk<=0:raise ValueError('Positive CAGRA itopk required')
  if cr and p['k_full_bit']>p['k_refine']:raise ValueError('CR-enabled k_full_bit must not exceed k_refine')
  if docs is not None and p['k_full_bit']>docs:raise ValueError('k_full_bit exceeds document capacity budget')
  result.append(dict(candidate_refinement_enabled=cr,k_refine_effective=cr,k_refine_semantics='partial-score TopK cap' if cr else 'ignored by no-CR; positive CLI placeholder',actual_CG_documents_bound=None if docs is None else [0,docs],num_refined_candidates='min(k_refine,actual_CG_documents)' if cr else 'actual_CG_documents',full_bit_target='min(k_full_bit,num_refined_candidates)',requested_k_full_bit=p['k_full_bit']))
 return result
