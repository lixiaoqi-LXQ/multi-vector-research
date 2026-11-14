"""Compare pinned official archives, Git tree blob identities, and local baseline/runtime trees."""
import hashlib,json,tarfile,os
from pathlib import Path
r=Path(__file__).parent

def digest(data): return hashlib.sha256(data).hexdigest()
def fingerprint(files):
    return digest(''.join(path+'\0'+digest(data)+'\n' for path,data in sorted(files.items())).encode())
def archive(path):
    blobs={};modes={};top=set()
    with tarfile.open(path,'r:gz') as t:
        for member in t.getmembers():
            components=member.name.split('/');top.add(components[0]);name='/'.join(components[1:])
            if not name:continue
            if member.isfile():blobs[name]=t.extractfile(member).read();modes[name]=member.mode
            elif member.issym():blobs[name]=member.linkname.encode();modes[name]=0o120000
    return blobs,modes,sorted(top)

def compare_local(files,path):
    root=Path(path);equal=[];missing=[];different=[]
    for name,data in sorted(files.items()):
        p=root/name
        if not p.exists() and not p.is_symlink():missing.append(name);continue
        actual=os.readlink(p).encode() if p.is_symlink() else p.read_bytes()
        if actual==data:equal.append(name)
        else:different.append({'path':name,'official_sha256':digest(data),'local_sha256':digest(actual),'official_bytes':len(data),'local_bytes':len(actual)})
    extras=[str(p.relative_to(root)) for p in root.rglob('*') if (p.is_file() or p.is_symlink()) and str(p.relative_to(root)) not in files]
    return {'path':str(root),'official_tracked_files':len(files),'equal_count':len(equal),'missing':missing,'different':different,'extra_untracked_files':extras,'baseline_match':not missing and not different,'tracked_file_fingerprint':fingerprint({name:(os.readlink(root/name).encode() if (root/name).is_symlink() else (root/name).read_bytes()) for name in files if (root/name).exists() or (root/name).is_symlink()})}

def check(name,cache):
    commit=json.loads((r/(name+'-main-commit.json' if name=='chimera' else name+'-fixed-commit.json')).read_text())
    tree=json.loads((r/(name+'-git-tree.json')).read_text())
    assert tree['sha']==commit['commit']['tree']['sha'] and not tree.get('truncated',False)
    files,modes,top=archive(r/(name+'-fixed.tar.gz'))
    wanted={x['path']:x for x in tree['tree'] if x['type']=='blob'}
    missing=sorted(set(wanted)-set(files));extra=sorted(set(files)-set(wanted));wrong=[]
    for path,item in wanted.items():
        if path not in files:continue
        data=files[path];blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if blob!=item['sha']:wrong.append({'path':path,'expected_git_blob':item['sha'],'actual_git_blob':blob})
    result={'repository':('iidyc/Chimera' if name=='chimera' else 'stanford-futuredata/ColBERT'),
            'commit_sha':commit['sha'],'tree_sha':tree['sha'],'commit_message':commit['commit']['message'],'commit_author_date':commit['commit']['author']['date'],
            'fixed_archive_sha256':digest((r/(name+'-fixed.tar.gz')).read_bytes()),'archive_top_directory':top,'archive_regular_or_symlink_count':len(files),'official_git_blob_count':len(wanted),'archive_git_blob_validation':{'missing':missing,'extra':extra,'wrong_blob_sha':wrong,'all_match':not missing and not extra and not wrong},'official_file_fingerprint_sha256':fingerprint(files),'cached_baseline':compare_local(files,cache)}
    assert result['archive_git_blob_validation']['all_match']
    (r/(name+'-official-file-manifest.json')).write_text(json.dumps({p:{'sha256':digest(data),'bytes':len(data),'git_blob_sha1':wanted[p]['sha'],'mode':wanted[p]['mode']} for p,data in sorted(files.items())},indent=2)+'\n')
    return result,files

chimera,ch_files=check('chimera','/home/lxq/graduate/ai-workspace/idea/multi-query/sources/Chimera-main')
plaid,pl_files=check('plaid','/home/lxq/graduate/ai-workspace/research/multivector/plaid/ColBERT-main')
local_archive=Path('/home/lxq/graduate/ai-workspace/research/multivector/plaid/upstream.tar.gz')
local,modes,top=archive(local_archive)
plaid['cached_download_archive']={'path':str(local_archive),'archive_sha256':digest(local_archive.read_bytes()),'archive_top_directory':top,'file_fingerprint_sha256':fingerprint(local),'same_source_files_as_fixed_commit':local==pl_files,'different_files':sorted(name for name in set(local)|set(pl_files) if local.get(name)!=pl_files.get(name))}
old_metadata=json.loads(Path('/home/lxq/graduate/ai-workspace/research/multivector/plaid/upstream-commit.json').read_text())
plaid['cached_commit_metadata']={'commit_sha':old_metadata['sha'],'tree_sha':old_metadata['commit']['tree']['sha'],'matches_current_official_fixed_commit':old_metadata['sha']==plaid['commit_sha'] and old_metadata['commit']['tree']['sha']==plaid['tree_sha']}
result={'GPU_used':False,'source_tree_modified':False,'chimera':chimera,'plaid':plaid}
(r/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
for name,item in [('chimera',chimera),('plaid',plaid)]:
 print(name,json.dumps({'commit_sha':item['commit_sha'],'tree_sha':item['tree_sha'],'git_blobs':item['official_git_blob_count'],'official_archive_blobs_match_git':item['archive_git_blob_validation']['all_match'],'cached_equal':item['cached_baseline']['equal_count'],'cached_missing':item['cached_baseline']['missing'],'cached_different':item['cached_baseline']['different'],'extra_count':len(item['cached_baseline']['extra_untracked_files']),'archive_same':item.get('cached_download_archive',{}).get('same_source_files_as_fixed_commit')},indent=2))
