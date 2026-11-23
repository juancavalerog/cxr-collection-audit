import json, sys, os
src = sys.argv[1]; out = sys.argv[2]
nb = json.load(open(src))
with open(out, 'w') as f:
    for i, c in enumerate(nb['cells']):
        s = ''.join(c['source'])
        f.write(f'\n##### CELL {i} [{c["cell_type"]}]\n{s}\n')
        if c['cell_type'] == 'code':
            for o in c.get('outputs', []):
                t = ''
                if 'text' in o: t = ''.join(o['text'])
                elif 'data' in o and 'text/plain' in o['data']: t = ''.join(o['data']['text/plain'])
                elif o.get('output_type') == 'error': t = 'ERROR ' + o.get('ename','') + ': ' + o.get('evalue','')
                if t:
                    if len(t) > 4000: t = t[:2000] + '\n...[recortado]...\n' + t[-2000:]
                    f.write(f'--- OUT ---\n{t}\n')
print(out, len(nb['cells']))
