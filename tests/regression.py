"""Integration checks against the supplied Python implementation. No extra packages."""
import argparse
import json
import math
import pathlib
import random
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ato

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--exe',default=str(ROOT/'build'/'atm_er2_optimizer.exe'))
    parser.add_argument('--gpu',action='store_true')
    args=parser.parse_args()
    scratch=ROOT/'work'/'tests'
    scratch.mkdir(parents=True,exist_ok=True)
    def run(*options, expected=0):
        p=subprocess.run([args.exe,*map(str,options)],capture_output=True,text=True,cwd=ROOT)
        if p.returncode!=expected:
            raise AssertionError(f'Exit {p.returncode}, expected {expected}: {p.stdout}\n{p.stderr}')
        return p.stdout
    run('--backend','cpu','--self-test','--threads',4,'--validation-layouts',256)
    rng=random.Random(919)
    masks=[ato.checkerboard_mask(),ato.spaced_3x3_mask(),ato.spaced_4x4_mask(),1,1<<24,ato.ALL_MASK]
    masks += [ato.random_mask(rng,1,49) for _ in range(18)]
    maximum_power=maximum_fuel=maximum_score=0.
    for i,mask in enumerate(masks):
        insertion,fill,variant=(0.,1.,1.) if i%2==0 else (.3,.6,1.2)
        reference=ato.simulate(mask,insertion=insertion,fill=fill,variant_efficiency=variant,
                               max_ticks=4500,min_ticks=1500,sample_ticks=500)
        prefix=scratch/f'python_{i}'
        run('--backend','cpu','--threads',1,'--math','exact','--evaluate',hex(mask),
            '--insertion',insertion*100,'--fill',fill,'--variant-efficiency',variant,'--output',prefix)
        result=json.loads(prefix.with_suffix('.json').read_text())['result']
        power=abs(result['power_fe_t']/reference.power_fe_t-1)
        fuel=abs(result['fuel_mb_t']/reference.fuel_mb_t-1)
        score=abs(result['efficiency_fe_per_mb']/reference.efficiency_fe_per_mb-1)
        maximum_power=max(maximum_power,power)
        maximum_fuel=max(maximum_fuel,fuel)
        maximum_score=max(maximum_score,score)
        assert max(power,fuel,score)<.002,(i,mask,power,fuel,score)
        assert result['rod_columns']==mask.bit_count()
    print(f'Python parity: {len(masks)} layouts; max power {maximum_power:.6%}, fuel {maximum_fuel:.6%}, efficiency {maximum_score:.6%}')
    common=['--backend','cpu','--threads',4,'--evaluations',512,'--seed',77,'--quiet']
    first=scratch/'deterministic_a';second=scratch/'deterministic_b'
    run(*common,'--output',first);run(*common,'--output',second)
    a=json.loads(first.with_suffix('.json').read_text())
    b=json.loads(second.with_suffix('.json').read_text())
    assert a['result']==b['result'],'Fixed-budget runs were nondeterministic'
    assert a['run']['candidate_evaluations']==512
    assert a['settings']['max_rods']==49,'Default search must allow the full footprint'
    tuning=['--agents',8,'--power-agents',2,'--elites',12,'--max-flips',49,'--mutation-percent',100,
            '--crossover-percent',75,'--random-percent',35,'--migration-generations',2,'--restart-generations',2,'--batch',3]
    run(*common,*tuning,'--output',first);run(*common,*tuning,'--output',second)
    a=json.loads(first.with_suffix('.json').read_text());b=json.loads(second.with_suffix('.json').read_text())
    assert a['result']==b['result'],'Multi-agent runs with sharing/restarts must be deterministic'
    assert a['run']['candidate_evaluations']==512 and a['run']['generations']==171
    no_result=scratch/'infeasible'
    run('--backend','cpu','--threads',2,'--evaluations',64,'--min-power',1e12,'--output',no_result,expected=3)
    assert not no_result.with_suffix('.json').exists()
    floor=scratch/'floor'
    run('--backend','cpu','--threads',4,'--evaluations',512,'--min-power',350000,'--output',floor)
    result=json.loads(floor.with_suffix('.json').read_text())
    assert result['result']['power_fe_t']>=350000 and result['run']['exact_cpu_verified']
    run('--threads',0,expected=1);run('--fill',0,expected=1)
    run('--insertion','nan',expected=1);run('--evaluate','0x2000000000000',expected=1)
    run('--sample-ticks',9000,expected=1);run('--unknown',expected=1)
    for option,value in [('--agents',0),('--agents',33),('--power-agents',5),('--elites',0),('--max-flips',0),
                         ('--mutation-percent',101),('--random-percent',101),('--restart-generations',1000001)]:
        run(option,value,expected=1)
    off=scratch/'fully_inserted'
    run('--backend','cpu','--threads',1,'--evaluations',8,'--insertion',100,'--output',off,expected=3)
    if args.gpu:
        run('--backend','cuda','--self-test','--threads',12,'--validation-layouts',1024)
        gpu=scratch/'gpu'
        run('--backend','cuda','--threads',12,'--evaluations',16384,'--min-power',350000,'--output',gpu)
        data=json.loads(gpu.with_suffix('.json').read_text())
        assert data['settings']['backend']=='cuda'
        assert data['result']['power_fe_t']>=350000 and data['run']['exact_cpu_verified']
        run('--backend','cuda','--threads',12,'--evaluations',16384,'--min-power',350000,'--output',first)
        assert json.loads(first.with_suffix('.json').read_text())['result']==data['result'],'Fixed-budget GPU run was nondeterministic'
        # The double GPU path is separately compiled; keep its validation sample bounded.
        run('--backend','cuda','--math','exact','--self-test','--threads',4,'--validation-layouts',32)
    print('PASS: deterministic budgets, power-floor enforcement, exact verification, CLI bounds, and output JSON')

if __name__=='__main__':
    main()
