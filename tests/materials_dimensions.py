"""Independent generalized reference checks; leave the original ato.py unchanged."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ato

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--exe',default=str(ROOT/'build/atm_er2_optimizer.exe'))
    parser.add_argument('--gpu',action='store_true')
    args=parser.parse_args()
    scratch=ROOT/'work/material_tests';scratch.mkdir(parents=True,exist_ok=True)
    presets=json.loads((ROOT/'data/moderators.json').read_text())['moderators']
    def run(*opts,expected=0):
        p=subprocess.run([args.exe,*map(str,opts)],cwd=ROOT,capture_output=True,text=True)
        assert p.returncode==expected,(p.returncode,p.stdout,p.stderr)
        return p.stdout
    assert len(run('--list-moderators').splitlines())==68
    rng=random.Random(919)
    shapes=[(3,5,4),(5,3,11),(1,1,1),(1,9,6),(9,1,8),(9,8,5),(16,12,32),(32,32,64)]
    max_error=0
    for i,m in enumerate(presets):
        w,d,h=shapes[i%len(shapes)]
        # A fresh geometry and material in the Python reference equations.
        ato.W,ato.D,ato.H=w,d,h;ato.CELL_COUNT=w*d;ato.REACTOR_VOLUME=w*d*h
        ato.INNER_SURFACE_AREA=2*(w*d+w*h+d*h)
        ato.OUTER_SURFACE_AREA=2*((w+2)*(d+2)+(w+2)*(h+2)+(d+2)*(h+2))
        ato.REACTOR_TO_COOLANT_COEFF=.6*ato.INNER_SURFACE_AREA
        ato.REACTOR_HEAT_LOSS_COEFF=.001*ato.OUTER_SURFACE_AREA
        ato.UNOBTAINIUM_ABSORPTION=m['absorption'];ato.UNOBTAINIUM_HEAT_EFFICIENCY=m['heat_efficiency']
        ato.UNOBTAINIUM_MODERATION=m['moderation'];ato.UNOBTAINIUM_CONDUCTIVITY=m['heat_conductivity']
        cells=rng.sample(range(w*d),min(12,w*d))
        mask=sum(1<<cell for cell in cells)
        insertion=.25 if i%3==0 else 0.;fill=.6 if i%4==0 else 1.
        ticks=4500
        reference=ato.simulate(mask,insertion=insertion,fill=fill,variant_efficiency=1.,max_ticks=ticks,min_ticks=ticks,sample_ticks=500)
        prefix=scratch/f'preset_{i}'
        common=['--width',w,'--depth',d,'--height',h,'--moderator',m['key'],'--evaluate',hex(mask),'--fixed-ticks',
                '--search-max-ticks',ticks,'--search-min-ticks',ticks,'--insertion',insertion*100,'--fill',fill,'--output',prefix]
        run('--backend','cpu','--math','exact','--threads',1,*common)
        data=json.loads(prefix.with_suffix('.json').read_text());result=data['result']
        for key,value in [('power_fe_t',reference.power_fe_t),('fuel_mb_t',reference.fuel_mb_t),('efficiency_fe_per_mb',reference.score)]:
            error=abs(result[key]-value)/max(abs(value),1e-9)
            max_error=max(max_error,error);assert error<.002,(m['key'],(w,d,h),key,error)
        assert data['settings']['geometry']==[w,d,h]
        assert data['settings']['moderator']['key']==m['key']
        assert result['rod_blocks']==len(cells)*h
        stored=result['mask'];assert (int(stored,16) if isinstance(stored,str) else stored)==mask
        assert len(result['layout'].splitlines())==d
        assert all(len(row.split())==w for row in result['layout'].splitlines())
    # A rod beyond bit 63 survives search, crossover, export and re-evaluation.
    for backend in (['cpu','cuda'] if args.gpu else ['cpu']):
        prefix=scratch/f'wide_{backend}'
        run('--backend',backend,'--width',9,'--depth',8,'--height',5,'--moderator','vibranium',
            '--evaluations',512,'--batch',256,'--threads',4,'--min-rods',65,'--max-rods',72,'--output',prefix)
        data=json.loads(prefix.with_suffix('.json').read_text());assert data['run']['exact_cpu_verified']
        assert data['result']['rod_columns']>=65 and int(data['result']['mask'],16).bit_count()==data['result']['rod_columns']
        if backend=='cuda':
            for w,d,h,material in [(3,5,4,'water'),(9,8,5,'vibranium'),(32,32,64,'graphite'),(1,1,1,'air')]:
                run('--backend','cuda','--width',w,'--depth',d,'--height',h,'--moderator',material,
                    '--self-test','--threads',4,'--validation-layouts',32,'--batch',32)
    run('--width',0,expected=1);run('--height',65,expected=1);run('--depth',33,expected=1)
    run('--moderator','missing',expected=1);run('--width',1,'--depth',1,'--max-rods',2,expected=1)
    run('--width',3,'--depth',3,'--evaluate','0x200',expected=1)
    run('--benchmark-only','--backend','cpu','--width',3,'--depth',5,'--threads',4,'--output',scratch/'benchmark')
    assert not (scratch/'benchmark.json').exists()
    print(f'PASS: all 68 moderator presets, 8 geometries, extended masks, CPU/GPU checks, bounds, benchmark-only; max Python error {max_error:.6%}')

if __name__=='__main__':main()
