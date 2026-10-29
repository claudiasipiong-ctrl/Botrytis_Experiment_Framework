"""Build standalone execution guide and horizontal diagram; no training or downloads."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from utils import bootstrap
bootstrap.install(__file__,ROOT)
import html
import re

STYLE='''body{margin:0;font:16px/1.65 system-ui,sans-serif;color:#193042;background:#f3f6f9}main{max-width:1180px;margin:auto;padding:32px}h1,h2,h3{line-height:1.25;color:#13334a}h1{font-size:2.3rem}h2{margin-top:2.4em;border-top:1px solid #cfdae4;padding-top:1em}a{color:#065a95}nav{position:sticky;top:0;background:#102f44;color:white;padding:12px 24px;z-index:1;display:flex;gap:20px;flex-wrap:wrap}nav a{color:#d5efff}pre{background:#112a3c;color:#edf6ff;padding:20px;border-radius:10px;overflow:auto;font:14px/1.65 ui-monospace,monospace}code{font-family:ui-monospace,monospace}p code,li code,td code{background:#e3edf3;padding:2px 5px;border-radius:4px}table{border-collapse:collapse;width:100%;font-size:.92rem;background:white}td,th{border:1px solid #d3dfe8;padding:10px;text-align:left;vertical-align:top}th{background:#dcecf5}.notice{border-left:5px solid #c88516;padding:18px;background:#fff3d7}details{margin:20px 0;padding:20px;background:white;border-radius:12px}summary{cursor:pointer;font-size:1.3rem;font-weight:bold}.diagram{overflow-x:auto;background:white;border:1px solid #d3dfe8}button{padding:10px 14px;margin:4px;border:1px solid #47677b;border-radius:8px;background:white;color:#193042;cursor:pointer}button:focus,button[aria-pressed=true]{outline:3px solid #2c89bc}svg text{font-family:system-ui,sans-serif}svg .lane{transition:opacity .2s}svg[data-filter] .lane{opacity:.18}svg .lane.active{opacity:1}@media print{nav,.controls{display:none}body{background:white}main{max-width:none;padding:0}pre{white-space:pre-wrap;color:black;background:#f0f0f0}details{break-inside:auto}svg{width:100%}}'''


def inline(s):
    s=html.escape(s)
    s=re.sub(r'`([^`]+)`',r'<code>\1</code>',s)
    s=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',s)
    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',s)


def render(text,offset=0):
    lines=text.splitlines();out=[];i=0
    while i<len(lines):
        line=lines[i]
        if line.startswith('```'):
            code=[];language=line[3:];i+=1
            while i<len(lines) and not lines[i].startswith('```'):code.append(lines[i]);i+=1
            out.append('<pre><code class="'+html.escape(language)+'">'+html.escape('\n'.join(code))+'</code></pre>')
        elif line.startswith('#'):
            n=len(line)-len(line.lstrip('#'));level=min(n+offset,6)
            out.append(f'<h{level}>'+inline(line[n:].strip())+f'</h{level}>')
        elif line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                row=[c.strip() for c in lines[i].strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',c or 'x') for c in row):rows.append(row)
                i+=1
            i-=1;out.append('<table>')
            for j,row in enumerate(rows):
                tag='th' if j==0 else 'td'
                out.append('<tr>'+''.join(f'<{tag}>'+inline(c)+f'</{tag}>' for c in row)+'</tr>')
            out.append('</table>')
        elif re.match(r'^[-*] ',line):
            out.append('<ul>')
            while i<len(lines) and re.match(r'^[-*] ',lines[i]):out.append('<li>'+inline(lines[i][2:])+'</li>');i+=1
            out.append('</ul>');i-=1
        elif re.match(r'^\d+\. ',line):
            out.append('<ol>')
            while i<len(lines) and re.match(r'^\d+\. ',lines[i]):out.append('<li>'+inline(re.sub(r'^\d+\. ','',lines[i]))+'</li>');i+=1
            out.append('</ol>');i-=1
        elif line.strip():out.append('<p>'+inline(line)+'</p>')
        i+=1
    return '\n'.join(out)


def page(title,body,script=''):
    nav='<nav><a href="GUIA_EJECUCION_3PCS_FINAL.html">Guía</a><a href="DIAGRAMA_FLUJO_3PCS_HORIZONTAL.html">Flujo</a><a href="COMANDOS_POWERSHELL_3PCS.md">Comandos</a><a href="PRUEBAS_Y_QA_PENDIENTE.md">Pruebas/QA</a><a href="CAMBIOS_Y_HALLAZGOS.md">Hallazgos</a></nav>'
    return '<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><style>'+STYLE+'</style></head><body>'+nav+'<main>'+body+'</main>'+script+'</body></html>'


def diagram():
    out=['<h1>Flujo horizontal · tres estaciones / siete modelos</h1><p class="notice">Diagrama de operación prevista. QA personal, pesos y experimento pendientes. RT-DETR-R50 bloqueado; no se certificaron tres GPUs. Los controles solo exploran el dibujo.</p>']
    models=[('PC-01','YOLOv8n'),('PC-01','YOLOv8l'),('PC-01','ResNet50'),('PC-02','YOLO26l'),('PC-02','Swin-Tiny'),('PC-03','RT-DETR-R50'),('PC-03','ViT-B/16')]
    out.append('<div class="controls"><button data-pc="all" aria-pressed="true">Todas las estaciones</button>'+''.join('<button data-pc="'+pc+'" data-model="'+m+'" aria-pressed="false">'+m+'</button>' for pc,m in models)+'</div><p id="detail" aria-live="polite">Un job local por GPU; paralelismo entre arquitecturas de estaciones distintas.</p>')
    svg=['<div class="diagram"><svg id="flow" viewBox="0 0 1600 1030" width="1600" role="img" aria-labelledby="diagram-title diagram-desc"><title id="diagram-title">Flujo de rondas supervisadas en tres PCs</title><desc id="diagram-desc">Base firmada común. Cada PC ejecuta baseline, HPO, robustez y candidato con revisiones humanas. Sincronización ZIP y transferencia manual. Test cerrado hasta freeze y autorización.</desc><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#36556b"/></marker></defs>']
    def box(x,y,w,h,label,small='',fill='#e5f1f8'):
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="#52758b"/><text x="{x+15}" y="{y+32}" font-size="19" font-weight="600">{html.escape(label)}</text><text x="{x+15}" y="{y+62}" font-size="15">{html.escape(small)}</text>')
    def arrow(x1,y1,x2,y2):svg.append(f'<path d="M{x1},{y1} L{x2},{y2}" stroke="#36556b" stroke-width="2" fill="none" marker-end="url(#arrow)"/>')
    box(25,22,1550,95,'Base común firmada: commit + dataset validado + split seed67 70/20/10 + tres folds internos','Python/entorno compatibles · identidad física explícita · pesos con procedencia/SHA · QA personal · SQLite en disco local',fill='#d6eee5')
    rows=[('PC-01','i21315','YOLOv8n · YOLOv8l · ResNet50'),('PC-02','i21314','YOLO26l · Swin-Tiny'),('PC-03','i21313','RT-DETR-R50 bloqueado · ViT-B/16')]
    for i,(pc,machine,names) in enumerate(rows):
        y=160+i*190;svg.append('<g class="lane active" data-pc="'+pc+'">')
        box(25,y,305,115,pc+' / '+machine,'Un trabajo GPU secuencial',fill='#d6eee5')
        svg.append(f'<text x="40" y="{y+94}" font-size="13">{html.escape(names)}</text>')
        box(360,y,200,92,'Baseline','Tres folds internos')
        box(680,y,200,92,'HPO','15 / 20 trials · 3 folds')
        box(1000,y,230,92,'Robustez','Top3 × 3 seeds × 3 folds')
        box(1360,y,215,92,'Candidato','Entrena 70 %, épocas fijas')
        arrow(330,y+46,355,y+46)
        for x1,x2 in [(560,675),(880,995),(1230,1355)]:
            arrow(x1,y+46,x2,y+46)
            svg.append(f'<rect x="{x1+10}" y="{y+103}" width="110" height="47" rx="7" fill="#ffedc7" stroke="#bd831c"/><text x="{x1+20}" y="{y+120}" font-size="13"><tspan x="{x1+20}">Revisión</tspan><tspan x="{x1+20}" dy="17">humana</tspan></text>')
        svg.append('</g>')
    box(25,750,480,96,'Snapshots ZIP + SHA-256 + recibos','Import ordinario = evidencia; no autoridad',fill='#dfe4f4')
    box(540,750,520,96,'Transferencia explícita de ownership pendiente','Oferta bloquea origen → recibo → confirmación → activar',fill='#ffedc7')
    box(1095,750,480,96,'Nueva ronda / repeat / comparación','IDs inmutables; no sobrescribe resultados previos',fill='#dfe4f4')
    svg.append('<path d="M1320,750 L1320,720 L440,720 L440,630" stroke="#36556b" stroke-dasharray="6 5" stroke-width="2" fill="none" marker-end="url(#arrow)"/>')
    box(25,900,430,95,'20 % externo predeclarado','Una ronda elegida por arquitectura',fill='#f1e9f8')
    box(505,900,480,95,'Cohorte → selección → integración','Crops predichos; siete candidatos completos',fill='#f1e9f8')
    box(1035,900,540,95,'Freeze + autorización personal → test 10 %','Test cerrado. Una nueva ronda no lo desbloquea.',fill='#ffedc7')
    arrow(455,946,500,946);arrow(985,946,1030,946);svg.append('</svg></div>');out+=svg
    out.append('<p>El reparto usa cargas ilustrativas 31,1 / 31,0 / 33,2 h, sin mediciones. El bloqueo R50 no impide ejecutar ViT en PC-03. Revisiones y nuevas rondas ocurren en el 70 %; una cohorte oficial incompleta no habilita el test.</p>')
    js='''<script>document.querySelectorAll('button[data-pc]').forEach(button=>button.addEventListener('click',()=>{const pc=button.dataset.pc;document.querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));const flow=document.querySelector('#flow');if(pc==='all')flow.removeAttribute('data-filter');else flow.setAttribute('data-filter',pc);document.querySelectorAll('.lane').forEach(l=>l.classList.toggle('active',pc==='all'||l.dataset.pc===pc));document.querySelector('#detail').textContent=pc==='all'?'Un job local por GPU; paralelismo entre arquitecturas de estaciones distintas.':button.dataset.model+' · '+pc+' · '+(button.dataset.model==='RT-DETR-R50'?'BLOQUEADO: backend/checkpoint pendiente de decisión.':'Baseline → revisión → HPO/bloques → revisión → robustez → revisión → candidato.');}));</script>'''
    return page('Diagrama horizontal Botrytis', ''.join(out),js)


def main():
    docs=ROOT/'docs'
    body='<h1>Guía de ejecución final · Botrytis</h1><p class="notice">Siete arquitecturas, tres PCs locales, rondas inmutables y revisiones humanas. Documento operativo para QA personal. No se ejecutó entrenamiento/HPO/test real; RT-DETR-R50 permanece bloqueado.</p>'
    for filename,title,opened in [('GUIA_ENTRENAMIENTO_ITERATIVO.md','Flujo iterativo y estados',True),('COMANDOS_POWERSHELL_3PCS.md','Instalación y comandos completos por PC',False),('ESTIMACION_TIEMPOS_ACTUALIZADA.md','Planificación y calibración personal',False)]:
        body+='<details'+(' open' if opened else '')+'><summary>'+title+'</summary>'+render((docs/filename).read_text(encoding='utf-8'),1)+'</details>'
    (docs/'GUIA_EJECUCION_3PCS_FINAL.html').write_text(page('Guía final Botrytis',body),encoding='utf-8')
    (docs/'DIAGRAMA_FLUJO_3PCS_HORIZONTAL.html').write_text(diagram(),encoding='utf-8')
    print('Generados guia y diagrama HTML estaticos; no se ejecutaron comandos de experimento.')

if __name__=='__main__':main()
