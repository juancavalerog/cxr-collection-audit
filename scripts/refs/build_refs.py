import csv, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from refs_data import R, CR, R1, DOICHK
base = os.path.expanduser('~/mnt/paper_new/refs')

# Tipos de entrada admitidos. 'dataset' y 'software' son tipos biblatex que pandoc lee de un .bib.
TIPOS = ('article', 'inproceedings', 'book', 'misc', 'dataset', 'software')
# El estilo elsevier-with-titles solo imprime container-title en los tipos que no son libro ni actas;
# por eso la etiqueta visible de preprints, conjuntos de datos y programas va en journaltitle.
ETIQUETADOS = ('misc', 'dataset', 'software')

# Nombres propios con solo la inicial en mayuscula que no deben pasar a minuscula.
PROPIOS = {'Fourier-based', 'Bayesian', 'Sinkhorn', 'National', 'Institutes', 'Health', 'Swin',
           "`Student's'", 'Extension', 'Part', 'I', 'II', 'III'}
PUNT_INI = '(["`'
PUNT_FIN = ').,:;?!"\''

def necesita_llaves(nucleo):
    if nucleo in PROPIOS:
        return True
    if any(c.isdigit() for c in nucleo) and any(c.isupper() for c in nucleo):
        return True
    for seg in nucleo.split('-'):
        if any(c.isupper() for c in seg[1:]):
            return True
        if len(seg) == 1 and seg.isupper() and '-' in nucleo:
            return True
    return False

def proteger(titulo):
    """Protege con llaves siglas, nombres de metodos y nombres propios; el resto queda
    a merced del estilo (pandoc pasa los titulos en ingles a mayuscula inicial de frase)."""
    out = []
    for tok in titulo.split(' '):
        if '{' in tok or not tok:
            out.append(tok); continue
        i, j = 0, len(tok)
        while i < j and tok[i] in PUNT_INI and tok[i:j] not in PROPIOS: i += 1
        while j > i and tok[j-1] in PUNT_FIN and tok[i:j] not in PROPIOS: j -= 1
        nucleo = tok[i:j]
        out.append(tok[:i] + '{' + nucleo + '}' + tok[j:] if nucleo and necesita_llaves(nucleo) else tok)
    return ' '.join(out)

def sede(v):
    return v.replace('npj ', '{npj} ')

def bib(r):
    key,tp,title,auth,year,venue,extra,doi,url,uv,est,nota = r
    assert tp in TIPOS, (key, tp)
    f = [('title', proteger(title)), ('author', auth), ('year', str(year))]
    if tp == 'article': f.append(('journal', sede(venue)))
    elif tp == 'inproceedings': f.append(('booktitle', sede(venue)))
    elif tp in ETIQUETADOS: f.append(('journaltitle', '{' + venue + '}'))
    for k,v in extra.items():
        if k == 'publisher' and ' and ' in v: v = '{' + v + '}'
        f.append((k, v))
    if doi: f.append(('doi', doi))
    if url: f.append(('url', url))
    elif doi: f.append(('url', 'https://doi.org/' + doi))
    return '@%s{%s,\n%s\n}\n' % (tp, key, ',\n'.join('  %s = {%s}' % (k, v) for k, v in f))

def plain(s):
    s = s.replace("{\\'o}","ó").replace("{\\~n}","ñ").replace('{\\"u}','ü').replace('{\\"o}','ö').replace("{\\'e}","é").replace("{\\'i}","í").replace("{\\'a}","á").replace("{\\c{c}}","ç").replace("{\\o}","ø").replace("{\\'y}","ý")
    return s.replace('{', '').replace('}', '')

with open(os.path.join(base,'refs.bib'),'w',encoding='utf-8') as fh:
    fh.write('% Generado por scripts/refs/build_refs.py (revision de referencias del 2026-11-21). Estado de cada entrada en verificacion_refs.csv\n\n')
    for r in R: fh.write(bib(r)+'\n')
with open(os.path.join(base,'verificacion_refs.csv'),'w',newline='',encoding='utf-8') as fh:
    w = csv.writer(fh)
    w.writerow(['clave','titulo','primer_autor','anio','sede','doi_o_url','url_de_verificacion','estado','nota',
                'doi_comprobado_2026_11_21','revision_r1_fecha','revision_r1_fuente','revision_r1_cambio'])
    for r in R:
        key,tp,title,auth,year,venue,extra,doi,url,uv,est,nota = r
        first = plain(auth.split(' and ')[0])
        if not uv: uv = CR+doi
        elif doi and 'query' in uv: uv = CR+doi
        r1 = R1.get(key, ('', '', ''))
        w.writerow([key,plain(title),first,year,plain(venue),doi if doi else url,uv,est,nota,
                    DOICHK.get(doi, '') if doi else '', r1[0], r1[1], r1[2]])
print('ok', len(R))
