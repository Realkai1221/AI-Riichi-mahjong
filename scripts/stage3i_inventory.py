"""Write-once Stage 3I file inventory and closed-ledger bookkeeping (stdlib only)."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,math,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
from stage3i import PHASE_LIMITS,FORMAL_NAMES,phase_for
ROOT=Path(__file__).resolve().parents[1];REPORTS=ROOT/'reports/stage3i';MANIFEST=REPORTS/'manifest.json';VERIFICATION=REPORTS/'inventory-verification.json'
AUX=('scripts/stage3i_inventory.py','scripts/package_a_stability.py','configs/stage3i.json','docs/STAGE3I_PLAN.md','docs/STAGE3I_REVIEW.md','docs/STAGE3I_PROPOSAL_REVIEW.md','docs/reviews/STAGE3I_PLAN_REV2_SOURCE.md','README.md','AGENTS.md','.gitignore','docs/STATUS.md','docs/DECISIONS.md','docs/IMPLEMENTATION_PLAN.md','pyproject.toml','requirements-macos-arm64.lock')
def require(test,message):
 if not test:raise ValueError(message)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def safe(root,relative):
 p=root/relative
 require(not Path(relative).is_absolute() and '..' not in Path(relative).parts and p.resolve().is_relative_to(root.resolve()) and not p.is_symlink(),'Unsafe inventory path')
 return p
def files(root):return {p for p in root.rglob('*') if p.is_file() and not p.name.endswith('.lock')}
def state():
 config=read(ROOT/'configs/stage3i.json');plan=read(REPORTS/'plan.json');data=Path(config['data_root']);ledger=read(data/'budget-ledger.json')
 require(data.is_absolute() and data.resolve()==data and not data.is_relative_to(ROOT),'Invalid external root')
 saved=read(REPORTS/'budget.json')
 require(all(saved.get(k)==v for k,v in ledger.items()) and ledger['identity']==sha(ROOT/'configs/stage3i.json')==plan['config_sha256'],'Ledger/config/plan mismatch')
 require(sha(plan['protocol_path'])==plan['protocol_sha256'],'Protocol bytes changed')
 for stem in ['reference','call']:require(sha(plan[stem+'_checkpoint'])==plan[stem+'_sha256'],'Historical checkpoint changed')
 require(sha(Path(plan['corpus_path'])/'manifest.json')==plan['corpus_manifest_sha256'] and sha(plan['hand_targets_path'])==plan['hand_targets_sha256'],'Historical corpus/target identity changed')
 require(sha(plan['cache_manifest_path'])==plan['cache_manifest_sha256'],'Feature cache manifest changed')
 total=0.;previous_end=0.;phases=dict.fromkeys(PHASE_LIMITS,0.);attempts=0;all_sources=set();old_sources=[];formal=[]
 for i,op in enumerate(ledger['operations']):
  name=op['name'];record=read(REPORTS/f'operation-{name}.json');phase=phase_for(name);elapsed=op['elapsed_seconds'];reserved=op['reserved_seconds']
  require(op['id']==i and op['status']=='completed' and not op['exceeded_reservation'] and 0<=elapsed<=reserved,'Unclosed/failed numerical lease')
  require(op['started_unix']+.05>=previous_end,'Overlapping numerical work');previous_end=op['started_unix']+elapsed
  require(reserved+config['reserve_seconds']+record['future_reserve_seconds']<=config['limit_seconds']-total+1e-8,'Protected future reserve violated')
  require(phases[phase]+reserved<=PHASE_LIMITS[phase]+1e-8,'Phase reservation exceeded')
  require(record['status']=='completed' and record['ledger_operation']==op and record['phase']==phase and record['config_sha256']==ledger['identity'],'Receipt mismatch')
  require(record['sampled_peak_tree_rss_bytes']<config['memory_bytes'] and record['artifact_bytes_peak']<config['artifact_bytes'],'Recorded resource cap reached')
  for relative,expected in record['sources'].items():
   all_sources.add(relative)
   if sha(safe(ROOT,relative))!=expected:
    require(name=='preflight-release-tests','Source changed after final release or formal operation: '+relative)
    old_sources.append(dict(operation=name,path=relative,sha256=expected,bytes_available=False))
  if name in FORMAL_NAMES:
   formal.append(name);require(all(record['sources'].get(k)==v for k,v in plan['sources'].items()),'Formal source binding differs')
  total+=elapsed;phases[phase]+=elapsed;attempts+=record['reserved_match_attempts']
 require(formal==['features-formal','training-formal','calibration-formal','evaluation-formal'],'Formal stage order differs')
 require(total==saved['charged_seconds'] and config['limit_seconds']-total==saved['remaining_seconds'],'Closed budget arithmetic mismatch')
 require(attempts==4360<=config['max_match_executions'],'Unexpected complete-match admission count')
 starts=list(data.glob('**/lifecycle/*.started.json'));finishes=list(data.glob('**/lifecycle/*.finished.json'))
 require(len(starts)==len(finishes)==4360,'Actual lifecycle count mismatch')
 require(len(list(REPORTS.glob('operation-*.json')))==len(ledger['operations']),'Unledgered operation receipt')
 for p in ['audit.json','replays.json','summary.json','posthoc-fractional-diagnostic.json']:require(read(REPORTS/p)['complete'] is True,'Final evidence incomplete')
 require(read(REPORTS/'historical-preservation.json')['valid'] is True,'Historical preservation failed')
 xml=ET.parse(REPORTS/'release-final.xml').getroot();suites=[xml] if xml.tag=='testsuite' else xml.findall('testsuite')
 counts={k:sum(int(s.get(k,'0')) for s in suites) for k in ['tests','failures','errors','skipped']}
 require(counts==dict(tests=153,failures=0,errors=0,skipped=0),'Final recorded tests did not all pass')
 account=read(REPORTS/'accounting.json');require(account['numerical_seconds']==total and account['operations']==len(ledger['operations']),'Published accounting is stale')
 roots={'project':ROOT,'external':data};project={safe(ROOT,p) for p in all_sources|set(AUX)}|files(REPORTS)
 for folder,pattern in [('scripts','*stage3i*.py'),('src/riichi_ai','stage3i*.py'),('tests','*stage3i*.py')]:project.update((ROOT/folder).glob(pattern))
 project-={MANIFEST,VERIFICATION};selected={'project':project,'external':files(data)}
 for scope,paths in selected.items():
  for p in paths:require(safe(roots[scope],p.relative_to(roots[scope]).as_posix()).is_file(),'Missing inventory member')
 size=sum(p.stat().st_size for p in files(data)|files(REPORTS) if p not in {MANIFEST,VERIFICATION})
 require(size+20_000_000<config['artifact_bytes'],'Insufficient final inventory disk allowance')
 accounting=dict(charged_seconds=total,limit_seconds=config['limit_seconds'],operations=len(ledger['operations']),phase_charged_seconds=phases,match_attempts=attempts,all_closed=True,final_test_counts=counts,final_tests_sha256=sha(REPORTS/'release-final.xml'),superseded_initial_preflight_sources=old_sources,artifact_bytes_excluding_inventory=size,policy_promotion=False)
 return roots,selected,accounting

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['write','verify']);p.add_argument('--output');args=p.parse_args();roots,selected,accounting=state()
 entries=[]
 for scope,paths in selected.items():
  for f in sorted(paths):entries.append(dict(scope=scope,path=f.relative_to(roots[scope]).as_posix(),bytes=f.stat().st_size,sha256=sha(f)))
 entries.sort(key=lambda e:(e['scope'],e['path']))
 base=dict(schema='stage3i-inventory-v1',roots={k:str(v) for k,v in roots.items()},file_count=len(entries),total_bytes=sum(e['bytes'] for e in entries),scope_counts=dict(Counter(e['scope'] for e in entries)),files=entries,accounting=accounting)
 if args.mode=='write':
  require(not MANIFEST.exists(),'Inventory is immutable; verify existing instead')
  base.update(created_utc=datetime.now(timezone.utc).isoformat(),working_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),exclusions=['This manifest and inventory-verification.json','External runtime lock files','Historical input files are separately hash-pinned and preserved'],limitations=['Integrity/accounting checks are not new numerical tests or full rule certification','Five superseded files from the first successful preflight test have hashes but no archived source bytes; all final-release and formal sources are retained','No learned-candidate formal games or policy promotion','Current working_head precedes result commit; listed hashes bind the final bytes'])
  with MANIFEST.open('x') as f:json.dump(base,f,indent=2,sort_keys=True);f.write('\n')
  result=dict(created=True,file_count=base['file_count'],total_bytes=base['total_bytes'],scope_counts=base['scope_counts'],manifest_sha256=sha(MANIFEST))
 else:
  existing=read(MANIFEST);require(all(existing[k]==v for k,v in base.items()),'Inventory membership/content/accounting mismatch')
  result=dict(valid=True,files_checked=base['file_count'],total_bytes=base['total_bytes'],scope_counts=base['scope_counts'],manifest_sha256=sha(MANIFEST),errors=[])
 if args.output:
  require(args.mode=='verify' and Path(args.output).resolve()==VERIFICATION,'Only verification output may be written')
  with VERIFICATION.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
