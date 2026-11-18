"""Stdlib-only immutable inventories and exact closed-ledger bookkeeping."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from stage3j import PHASE_LIMITS, phase_for

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/stage3j'
MANIFEST=REPORTS/'manifest.json'
VERIFY=REPORTS/'inventory-verification.json'
OLD_COMMIT='17fc2e5900a1e9102b6930f01ca3cb5b7774a959'

def require(test,message):
    if not test:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()
def write(path,value):
    with Path(path).open('x') as handle:json.dump(value,handle,indent=2,sort_keys=True);handle.write('\n')
def safe(root,name):
    path=root/name
    require(not Path(name).is_absolute() and '..' not in Path(name).parts and path.resolve().is_relative_to(root.resolve())
            and not path.is_symlink(),'Unsafe inventory path')
    return path
def files(root):return {p for p in root.rglob('*') if p.is_file() and not p.name.endswith('.lock')}

def history():
    source=ROOT/'reports/stage3i/manifest.json'
    require(sha(source)=='644e8291397a0e0ad2ce0d8d90fd481aa4b8bf05f9886e916df63843e6fbf72f','Historical inventory changed')
    old=read(source);roots={k:Path(v) for k,v in old['roots'].items()};git=[]
    for item in old['files']:
        path=safe(roots[item['scope']],item['path'])
        if path.is_file() and sha(path)==item['sha256'] and path.stat().st_size==item['bytes']:continue
        require(item['scope']=='project','Historical external evidence changed: '+str(path))
        raw=subprocess.check_output(['git','show',OLD_COMMIT+':'+item['path']],cwd=ROOT)
        require(len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'],'Historical Git bytes changed')
        git.append(item['path'])
    result=dict(valid=True,files_checked=len(old['files']),total_bytes=old['total_bytes'],manifest_sha256=sha(source),
        historical_commit=OLD_COMMIT,shared_files_verified_from_git=git,
        interpretation='Original historical bytes preserved; shared current documentation may evolve')
    write(REPORTS/'historical-preservation.json',result)
    return result

def state():
    config=read(ROOT/'configs/stage3j.json');data=Path(config['data_root']);plan=read(REPORTS/'plan.json')
    ledger=read(data/'budget-ledger.json');snapshot=read(REPORTS/'budget.json')
    require(all(snapshot.get(k)==v for k,v in ledger.items()) and ledger['identity']==sha(ROOT/'configs/stage3j.json')==plan['config_sha256'],
            'Ledger/config/plan mismatch')
    require(sha(plan['protocol_path'])==plan['protocol_sha256'],'Protocol changed')
    for item in plan['dependencies']:require(sha(item['path'])==item['sha256'],'Historical dependency changed')
    total=0.;previous_end=0.;phases=dict.fromkeys(PHASE_LIMITS,0.);attempts=0;source_names=set();superseded=[]
    status=Counter();peak=0
    for i,op in enumerate(ledger['operations']):
        name=op['name'];receipt=read(REPORTS/f'operation-{name}.json');phase=phase_for(name)
        elapsed,reserved=op['elapsed_seconds'],op['reserved_seconds']
        require(op['id']==i and op['status'] in ('completed','failed') and elapsed is not None,'Unclosed numerical lease')
        require(not op['exceeded_reservation'] and 0<=elapsed<=reserved,'Numerical deadline exceeded')
        require(op['started_unix']+.05>=previous_end,'Overlapping numerical operations');previous_end=op['started_unix']+elapsed
        require(reserved+config['reserve_seconds']+receipt['future_reserve_seconds']<=config['limit_seconds']-total+1e-8,'Future reserve violated')
        require(phases[phase]+reserved<=PHASE_LIMITS[phase]+1e-8,'Phase cap violated')
        require(receipt['status']==op['status'] and receipt['ledger_operation']==op and receipt['phase']==phase,'Receipt mismatch')
        require(receipt['sampled_peak_tree_rss_bytes']<config['memory_bytes'] and receipt['artifact_bytes_peak']<config['artifact_bytes'],'Resource ceiling reached')
        peak=max(peak,receipt['sampled_peak_tree_rss_bytes'])
        for relative,expected in receipt['sources'].items():
            source_names.add(relative)
            current=safe(ROOT,relative)
            if sha(current)!=expected:
                require(phase=='preflight','Post-freeze numerical source mutation: '+relative)
                superseded.append(dict(operation=name,path=relative,sha256=expected,bytes_archived=True))
            require(sha(data/'source-archive'/(expected+'.bin'))==expected,'Missing numerical source archive')
        if name=='evaluation-formal':
            require(op['status']=='completed' and all(receipt['sources'].get(k)==v for k,v in plan['sources'].items()),'Formal execution failed or differs')
        status[op['status']]+=1;total+=elapsed;phases[phase]+=elapsed;attempts+=receipt['reserved_match_attempts']
    require(total==snapshot['charged_seconds'] and config['limit_seconds']-total==snapshot['remaining_seconds'],'Budget arithmetic mismatch')
    require(attempts<=config['max_match_executions'],'Match reservation cap')
    starts=list(data.glob('**/lifecycle/*.started.json'));finishes=list(data.glob('**/lifecycle/*.finished.json'))
    require(len(starts)==len(finishes)==8288 and attempts==8288,'Unexpected game executions/reservations')
    require(len(list(REPORTS.glob('operation-*.json')))==len(ledger['operations']),'Unledgered receipt')
    for name in ('audit.json','replays.json','summary.json','evaluation.json'):
        require(read(REPORTS/name)['complete'] is True,'Incomplete final evidence')
    require(read(REPORTS/'historical-preservation.json')['valid'] is True,'Historical preservation failed')
    xml=ET.parse(REPORTS/'release-final.xml').getroot();suites=[xml] if xml.tag=='testsuite' else xml.findall('testsuite')
    counts={k:sum(int(s.get(k,'0')) for s in suites) for k in ('tests','failures','errors','skipped')}
    require(counts['tests']>0 and counts['failures']==counts['errors']==counts['skipped']==0,'Final tests failed or skipped')
    account=dict(numerical_seconds=total,limit_seconds=config['limit_seconds'],operations=len(ledger['operations']),
        operation_statuses=dict(status),phase_seconds=phases,reserved_match_attempts=attempts,complete_match_executions=len(finishes),
        formal_games=8192,preflight_games=64,replay_games=32,new_optimizer_updates=0,new_training_matches=0,
        sampled_peak_tree_rss_bytes=peak,final_test_counts=counts,superseded_preflight_sources=superseded,
        policy_promotion=False,all_leases_closed=True)
    project={safe(ROOT,x) for x in source_names}|files(REPORTS)
    project.update(ROOT/x for x in ('AGENTS.md','README.md','.gitignore','docs/STATUS.md','docs/DECISIONS.md','docs/IMPLEMENTATION_PLAN.md',
                                    'docs/STAGE3J_PLAN.md','docs/STAGE3J_REVIEW.md','requirements-macos-arm64.lock'))
    project.update((ROOT/'scripts').glob('stage3j*.py'))
    project-={MANIFEST,VERIFY}
    return {'project':ROOT,'external':data},{'project':project,'external':files(data)},account

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('history','account','write','verify'));args=parser.parse_args()
    if args.mode=='history':print(json.dumps(history()));return
    roots,members,account=state()
    if args.mode=='account':write(REPORTS/'accounting.json',account);print(json.dumps(account));return
    require(read(REPORTS/'accounting.json')==account,'Published accounting is stale')
    entries=[]
    for scope,paths in members.items():
        for path in sorted(paths):
            relative=path.relative_to(roots[scope]).as_posix();safe(roots[scope],relative)
            entries.append(dict(scope=scope,path=relative,bytes=path.stat().st_size,sha256=sha(path)))
    entries.sort(key=lambda x:(x['scope'],x['path']))
    manifest=dict(schema='stage3j-inventory-v1',roots={k:str(v) for k,v in roots.items()},files=entries,
                  file_count=len(entries),total_bytes=sum(e['bytes'] for e in entries),scope_counts=dict(Counter(e['scope'] for e in entries)),accounting=account)
    if args.mode=='write':
        manifest.update(created_utc=datetime.now(timezone.utc).isoformat(),working_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            exclusions=['This manifest and inventory-verification.json','External runtime lock files','Historical dependencies separately hash-pinned'],
            limitations=['Same-engine evidence checks, not full independent rule certification','No policy promotion','All numerical source releases retain exact archived bytes'])
        write(MANIFEST,manifest)
    else:
        existing=read(MANIFEST);require(all(existing.get(k)==v for k,v in manifest.items()),'Inventory changed')
        write(VERIFY,dict(valid=True,files_checked=len(entries),total_bytes=manifest['total_bytes'],manifest_sha256=sha(MANIFEST),errors=[]))
    print(json.dumps(dict(mode=args.mode,files=len(entries),total_bytes=manifest['total_bytes'],manifest_sha256=sha(MANIFEST))))

if __name__=='__main__':main()
