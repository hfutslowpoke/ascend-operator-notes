"""Create reproducible empirical roofline references for the five owned points."""
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
calibration = json.loads((HERE / 'results/roofline-calibration-20261105.json').read_text())
timing = json.loads((HERE / 'results/r185-owned5-roofline-timing.json').read_text())
assert calibration['status'] == 'finished' and len(timing['records']) == 5
primitives = {r['case']['name']: r for r in calibration['records']}
cube_tops = primitives['calib_cube_2048']['achieved_tops']
gelu_geps = primitives['calib_gelu_128']['achieved_gelu_giga_elements_per_second']
hot_gbps = primitives['calib_copy_8m_32']['read_plus_write_gbps']
stream_gbps = primitives['calib_copy_256m_4']['read_plus_write_gbps']
observations = {r['case']: r['variants']['base'] for r in timing['records']}
points = [('C2', 'pub_c2', 4, 3, 5, 3, 'official3'),
          ('R1', 'r1_sample', 64, 128, 128, 1, 'r1_sample'),
          ('I1', 'main_i1_sample', 1, 1, 1, 5, 'sh_i1'),
          ('I4', 'ex_i4_sample', 4093, 1019, 765, 8, 'ex_i4_sample'),
          ('I5', 'ex_i5_sample', 9217, 513, 385, 24, 'ex_i5_sample')]
rows = []
for point, name, m, k, n, e, directory in points:
    input_dir = HERE / 'cases' / directory / 'input'
    group_file = input_dir / 'group_list.bin'
    lengths = np.diff(np.r_[0, np.fromfile(group_file, np.int64)])
    active = int(np.count_nonzero(lengths))
    tiles = int(sum((int(x) + 127) // 128 for x in lengths))
    ops = 2*m*k*n
    footprint = m*k + e*k*n + 4*e*n + 8*e + 2*m*n
    onepass = m*k + active*k*n + 4*active*n + 8*e + 2*m*n
    warm = observations[name]['warm_median_us']
    logical = None
    if point == 'R1':
        logical = dict(x_read=m*k, b_read=4*k*n, scale_read=8*n*4,
                       c_write=4*m*n, c_read=4*m*n, y_write=2*m*n)
    elif point == 'I4':
        pitch = math.ceil(n/32)*32
        logical = dict(b_stage_read=e*k*n, b_stage_write=e*k*pitch,
                       cube_x_read=math.ceil(n/256)*m*k, cube_b_read=tiles*k*pitch,
                       c_write=4*m*pitch, c_read=4*m*n, scale_read=2*tiles*n*4,
                       y_write=2*m*n)
    elif point == 'I5':
        nt = math.ceil(n/256)
        main_k = k//128*128
        # Payload approximation, excluding hardware transaction rounding,
        # internal Matmul reloads and group-table scalar/cache transactions.
        logical = dict(cube_x_read=nt*m*main_k, cube_b_read=tiles*main_k*n,
                       tail_a_read=nt*m*(k-main_k), tail_b_read=2*tiles*(k-main_k)*n,
                       c_write=4*m*n, c_read=4*m*n, scale_read=2*tiles*n*4,
                       y_write=2*m*n)
    algorithm_bytes = sum(logical.values()) if logical else None
    cube_ref = ops/(cube_tops*1e6)
    gelu_ref = m*n/(gelu_geps*1e3)
    transfer_ref = algorithm_bytes/(hot_gbps*1e3) if algorithm_bytes else None
    ref_max = max(cube_ref, gelu_ref, transfer_ref or 0)
    row = dict(point=point, shape=[m,k,n,e], input_case=name,
               input_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(input_dir.iterdir())},
               active_experts=active, m128_tiles=tiles, effective_integer_ops=ops,
               tensor_footprint_bytes=footprint, onepass_active_payload_reference_bytes=onepass,
               logical_payload_components=logical, logical_payload_estimate_bytes=algorithm_bytes,
               arithmetic_intensity_tensor_ops_per_byte=ops/footprint,
               arithmetic_intensity_logical_ops_per_byte=ops/algorithm_bytes if algorithm_bytes else None,
               measured_warm_us=warm, measured_cold_us=[x['cold_us'] for x in observations[name]['runs']],
               measured_effective_tops=ops/(warm*1e6),
               idealized_cube_reference_us=cube_ref, idealized_gelu_reference_us=gelu_ref,
               hot_copy_logical_payload_reference_us=transfer_ref,
               optimistic_max_resource_reference_us=ref_max,
               measured_over_reference_ratio=warm/ref_max,
               caveat=('Sub-microsecond throughput references do not model launch, scalar load dependencies or usable core parallelism.'
                       if point in ['C2','I1','R1'] else
                       'Optimistic reference omits phase ordering, exact ND2NZ/Fixpipe paths, padding compute, partial-tile balance, synchronization and casts/scales.'))
    rows.append(row)

out_dir = HERE / 'results/roofline-owned5-20261105'
out_dir.mkdir(exist_ok=True)
report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
              carrier=timing['variants']['base'],
              calibration_file=str(HERE / 'results/roofline-calibration-20261105.json'),
              timing_file=str(HERE / 'results/r185-owned5-roofline-timing.json'),
              measured_references=dict(cube_tops=cube_tops, gelu_giga_elements_per_second=gelu_geps,
                                       hot_small_working_set_read_write_gbps=hot_gbps,
                                       cache_overflow_stream_read_write_gbps=stream_gbps,
                                       l2_bytes=176160768, aic=20, aiv=40,
                                       empty_aiv_warm_us=[1.65,1.08]),
              interpretation=[
                  'These are measured primitive reference rates, not proven architectural maxima or attainable operator latency lower bounds.',
                  'Copy rates count both read and write. Small working set16MiB is below168MiB L2; stream working set512MiB is larger. Neither is asserted to be pure HBM traffic.',
                  'GELU primitive is chained Horner FP32 Exp/Div with final FP16 cast once per launch, and omits per-output INT32 cast, scale and GM input.',
                  'Tensor footprint, one-pass payload, logical repeated payload and physical memory transactions are different quantities.',
                  'Logical estimates sum Cube reads, C write/read and Vector output/scale with no cache-transaction model. They are not measured HBM bytes.',
                  'UB attribute returned zero without a recorded status; unsupported/unknown, not zero physical UB.',
                  'I4 historical PMU replay shows Cube19.8%, AIC MTE252.9%, Fixpipe31.3%; overlapping activity ratios are not additive walltime shares.',
                  'Empty launch samples varied with width and task order. Do not subtract a universal fixed launch constant.',
                  'All timings are local msprof Task Duration. Public platform latency and hidden aggregate scoring remain independent observations.'
              ], rows=rows)
(out_dir / 'model.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
fields = [k for k,v in rows[0].items() if not isinstance(v,(dict,list))]
with (out_dir/'model.csv').open('w', newline='') as f:
    writer=csv.DictWriter(f, fieldnames=fields, extrasaction='ignore');writer.writeheader();writer.writerows(rows)

plt.rcParams.update({'figure.dpi':140,'font.size':10})
fig, ax=plt.subplots(figsize=(9,5.5))
x=np.geomspace(.01, 10000, 800)
ax.loglog(x, np.minimum(cube_tops, hot_gbps*x/1000), label='Hot small-working-set copy + Cube reference')
ax.loglog(x, np.minimum(cube_tops, stream_gbps*x/1000), '--', label='Cache-overflow stream copy + Cube reference')
for row in rows:
    ai=row['arithmetic_intensity_tensor_ops_per_byte'];tops=row['measured_effective_tops']
    ax.scatter(ai,tops,s=45,color='tab:blue')
    ax.annotate(row['point'],(ai,tops),xytext=(5,6),textcoords='offset points')
    logical_ai=row['arithmetic_intensity_logical_ops_per_byte']
    if logical_ai:
        ax.plot([logical_ai,ai],[tops,tops],color='gray',alpha=.65,lw=1)
        ax.scatter(logical_ai,tops,marker='x',color='tab:red')
ax.set(xlabel='Integer ops / byte (dot: tensor footprint, cross: estimated repeated payload)',
       ylabel='Effective INT8 TOPS (MAC = 2 ops)',title='Owned points: empirical reference roofs and measured operator throughput')
ax.grid(True,which='both',alpha=.15);ax.legend(loc='lower right',fontsize=8)
fig.text(.02,.01,'Reference curves omit GELU, launch, layout paths and synchronization; cache levels are not HBM claims.',fontsize=8)
fig.tight_layout(rect=[0,.035,1,1]);fig.savefig(out_dir/'roofline.png');fig.savefig(out_dir/'roofline.svg');plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11,4.6))
for ax,selected,title in [(axes[0],rows[:3],'Small points: launch/dependency costs matter'),
                          (axes[1],rows[3:],'Larger points: optimistic resource references')]:
    xx=np.arange(len(selected));w=.18
    series=[('Measured local warm','measured_warm_us'),('Cube reference','idealized_cube_reference_us'),
            ('GELU reference','idealized_gelu_reference_us'),('Hot-copy payload reference','hot_copy_logical_payload_reference_us')]
    for j,(label,key) in enumerate(series):
        ax.bar(xx+(j-1.5)*w,[r[key] or 0 for r in selected],w,label=label)
    ax.set_xticks(xx,[r['point'] for r in selected]);ax.set_title(title,fontsize=10);ax.set_ylabel('Microseconds');ax.grid(axis='y',alpha=.15)
axes[1].legend(fontsize=8)
fig.text(.015,.01,'References are not additive and are not proven latency floors. C2/I1 scalar traffic reference omitted.',fontsize=8)
fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(out_dir/'time-references.png');fig.savefig(out_dir/'time-references.svg');plt.close(fig)
print(json.dumps({r['point']:{k:r[k] for k in ['measured_warm_us','idealized_cube_reference_us','idealized_gelu_reference_us','hot_copy_logical_payload_reference_us']} for r in rows},indent=2))
print(out_dir)
