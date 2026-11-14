from pathlib import Path
import difflib
root=Path(__file__).resolve().parent
src=Path('/home/lxq/graduate/ai-workspace/research/multivector/chimera/run_search_grid.py');s=src.read_text();(root/'run_search_grid.base.py').write_text(s)
helper='''def check_derived_build_provenance(spec, buildmeta, index_meta):
 operation='CPU requantize from fixed 4-bit index to 8-bit'
 derived=buildmeta.get('fresh_8bit_build') is False or buildmeta.get('operation')==operation
 if not derived:return
 need(spec.get('allow_derived_index') is True,'Derived index requires explicit allow_derived_index')
 need(buildmeta.get('operation')==operation and buildmeta.get('fresh_8bit_build') is False,'Invalid derived operation identity')
 need(buildmeta.get('requantize_tool_sha256')==spec.get('approved_requantize_tool_sha256') and isinstance(spec.get('approved_requantize_tool_sha256'),str),'Unapproved requantize tool identity')
 need(buildmeta.get('qualification_receipt_sha256')==spec.get('approved_requantize_qualification_sha256') and isinstance(spec.get('approved_requantize_qualification_sha256'),str),'Unapproved requantization qualification')
 need(buildmeta.get('index_identity')==index_meta,'Derived metadata/index identity mismatch')
 rp=Path(buildmeta['requantize_receipt']);need(sha(rp)==buildmeta['requantize_receipt_sha256'],'Derived receipt SHA mismatch')
 r=json.loads(rp.read_text());need(r['operation']==operation and r['fresh_8bit_build'] is False and r['validation']['all_rows_bitwise_source_verified'] is True and r['validation']['gpu_used'] is False,'Incomplete CPU requantization evidence')
 need(r['source_build_receipt_sha256']==buildmeta['source4_build_receipt_sha256'] and r['tool_sha256']==buildmeta['requantize_tool_sha256'],'Derived/source receipt binding mismatch')
 for name,detail in index_meta['files'].items():need(detail['sha256']==r['outputs'][name]['sha256'] and detail['size']==r['outputs'][name]['bytes'],'Derived output payload mismatch: '+name)
 qp=Path(buildmeta['qualification_receipt']);need(sha(qp)==buildmeta['qualification_receipt_sha256'],'Qualification receipt changed')
 q=json.loads(qp.read_text());need(q['status']=='PASS_CPU_BYTE_QUALIFICATION' and q['binary_sha256']==r['tool_sha256'],'Qualification/tool mismatch')

'''
s=s.replace('def cpu_set(text):',helper+'def cpu_set(text):')
needle="  need(buildmeta['docs']==nd and buildmeta['tokens']==index_meta['N'] and buildmeta['total_bits']==index_meta['total_bits'],'Build provenance dimensions/bitwidth mismatch')"
assert needle in s;s=s.replace(needle,needle+'\n if buildmeta is not None:check_derived_build_provenance(spec,buildmeta,index_meta)')
(root/'run_search_grid.derived.py').write_text(s)
(root/'run-search-derived-acceptance.patch').write_text(''.join(difflib.unified_diff(src.read_text().splitlines(True),s.splitlines(True),fromfile='a/run_search_grid.py',tofile='b/run_search_grid.py')))
