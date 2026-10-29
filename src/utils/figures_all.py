"""Source-backed catalogue. Missing observations produce pending rows, never invented lines."""
from __future__ import annotations
import csv,json,hashlib,math
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def catalog():return json.loads((ROOT/'configs/figure_catalog.json').read_text(encoding='utf-8'))['items']
META=('dataset','split','phase','metric','unit','denominator','configuration','semantic_version','source_sha256')
class InsufficientData(ValueError):pass
def load_source(spec,index,allow_test=False):
    import pandas as pd
    item=index.get(spec['table'])
    if not item:raise InsufficientData('Falta '+spec['required_artifact']+' y descriptor en --source-index')
    if any(k not in item or item[k] in ('',None) for k in META):raise InsufficientData('Descriptor incompleto: '+','.join(META))
    if item['semantic_version']!='botrytis-physical-fruit-pairing-v4':raise InsufficientData('Fuente obsoleta; requiere semantica v4')
    if str(item['split']).strip().lower() not in ("train","validation","inner_validation","dataset_inventory","synthetic","test"):raise InsufficientData("Split no declarado/ambiguo")
    if item['split']=='test' and not allow_test:raise InsufficientData('Test bloqueado en generador; necesita freeze, release manual y --allow-final-test')
    p=Path(item['path']);p=p if p.is_absolute() else ROOT/p
    if not p.is_file() or sha(p)!=item['source_sha256']:raise InsufficientData('Fuente ausente/alterada; revisar hash '+str(p))
    df=pd.read_csv(p)
    if item.get('is_estimated') or item.get('synthetic') or ('is_estimated' in df and df['is_estimated'].astype(str).str.lower().isin(['true','1','yes']).any()):raise InsufficientData('Fuentes ESTIMADAS/ILUSTRATIVAS no son resultados oficiales')
    if not len(df):raise InsufficientData('Fuente sin observaciones')
    for c in ('split','phase','dataset','configuration'):
        if c in df and df[c].nunique(dropna=False)>1 and item["split"]!="dataset_inventory":raise InsufficientData('No agregar estimadores incompatibles: '+c)
    if any(k in (spec['x'],spec['y']) for k in ('latency_ms',)) or spec['id']=='pipeline_latency':
        required=('hardware','batch','resolution','warmup','cuda_sync','precision','include_disk_io')
        protocol=item.get('latency_protocol',{})
        if any(k not in protocol for k in required) or protocol.get('cuda_sync') is not True:raise InsufficientData('Latencias sin protocolo comun de hardware/batch/resolucion/warmup/CUDA')
        if 'hardware' in df and df['hardware'].nunique()>1:raise InsufficientData('Hardware distinto: latencias no comparables')
    if spec['id'] in ('difficulty','density','visual_conditions') and not item.get('strata_definition_source'):raise InsufficientData('Faltan estratos predeclarados/metadatos fiables')
    return df,item,p
def conceptual(spec,out):
    import html
    sid=spec['id']
    if sid=='gt_matching':boxes=['0 = frutos fisicos','1 = sospechosos','Maxima IoU global 1:1','Anomalias: bloquear/revisar','Solo crop clase 0; etiqueta 0↔1']
    elif sid=='pipeline_horizontal':boxes=['Imagen → detector fruit','Bboxes predichas','Crops → clasificador','P(con_signos) + umbral','GT 1:1 → errores separados']
    elif sid=='cib_reference':boxes=['Entrada','DW Conv → PW Conv','DW Conv (bloque central)','PW Conv → DW Conv','Residual opcional']
    else:boxes=['Entrada + posicion','Proyecciones Q, K, V','Softmax(QKᵀ / √d)','Multiples cabezas → proyeccion','Residual / normalizacion (segun modelo)']
    title=spec['title'];note='ESQUEMA METODOLOGICO; IoU tecnico inicial >=0.50 configurable en pairing.min_iou; clase 2 auxiliar; sin mediciones' if spec['kind']=='diagram' else 'REFERENCIA ARQUITECTONICA NO IMPLEMENTADA como bloque medido. CIB no es C2PSA; MHSA no sustituye AIFI/C2PSA.'
    source='src/utils/pairing.py; src/utils/pipeline_eval.py' if spec['kind']=='diagram' else 'https://arxiv.org/abs/2405.14458 (CIB); https://arxiv.org/abs/1706.03762 (MHSA)'
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1560" height="360" viewBox="0 0 1560 360" role="img" aria-labelledby="title desc"><title id="title">{html.escape(title)}</title><desc id="desc">{html.escape(note)}</desc><rect width="1560" height="360" fill="white"/><text x="32" y="44" font-family="Arial" font-size="23">{html.escape(title)}</text>']
    for i,b in enumerate(boxes):
        x=30+i*305;svg.append(f'<rect x="{x}" y="110" width="280" height="85" rx="8" fill="#e4f2f3" stroke="#195a68"/><text x="{x+12}" y="155" font-family="Arial" font-size="14">{html.escape(b)}</text>')
        if i<4:svg.append(f'<text x="{x+282}" y="158" font-size="25">→</text>')
    svg.append(f'<text x="30" y="252" font-family="Arial" font-size="16">{html.escape(note)}</text><text x="30" y="285" font-family="Arial" font-size="14">N = no aplica · Fuente: {html.escape(source)}</text></svg>')
    p=out/(sid+'.svg');p.write_text(''.join(svg),encoding='utf-8');return [p],source
def numeric(df,columns):
    import pandas as pd,numpy as np
    for c in columns:
        if c not in df:raise InsufficientData('Campo requerido: '+c)
        if not pd.api.types.is_numeric_dtype(df[c]) or not np.isfinite(df[c]).all():raise InsufficientData('Campo no numerico/no finito: '+c)
def render(spec,df,item,p,out,allow_pixels=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    kind,x,y,g=spec['kind'],spec['x'],spec['y'],spec['group']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    colors=['#176e82','#c28a20','#9c4a7b','#5d7336','#b95c35']
    if kind.startswith('two_panel'):fig,axes=plt.subplots(1,2,figsize=(15,6));ax=axes[0]
    else:fig,ax=plt.subplots(figsize=(10,6))
    try:
        if kind=='images':
            if not allow_pixels:raise InsufficientData('Requiere --allow-pixels personal; imagenes/Grad-CAM verificadas y sin test')
            from PIL import Image
            if not {'image_path','annotation_json','image_sha256'}.issubset(df):raise InsufficientData('Requiere image_path, image_sha256, annotation_json (cajas, signos, errores)')
            plt.close(fig);n=min(len(df),8);fig,axes=plt.subplots(math.ceil(n/4),min(n,4),figsize=(16,4*math.ceil(n/4)),squeeze=False)
            for axis,(_,r) in zip(axes.flat,df.head(n).iterrows()):
                path=Path(r.image_path);path=path if path.is_absolute() else ROOT/path
                if sha(path)!=r.image_sha256:raise InsufficientData('Imagen alterada: '+str(path))
                axis.imshow(Image.open(path).convert('RGB'))
                boxes=json.loads(r.annotation_json)
                for box in boxes:
                    from matplotlib.patches import Rectangle
                    a,b,c,d=box['xyxy'];axis.add_patch(Rectangle((a,b),c-a,d-b,fill=False,edgecolor=colors[int(box.get('class_id',0))%5],label=box.get('label','anotacion')))
                axis.set_title(str(r.get('image_id','ejemplo')));axis.set_xlabel('x (px)');axis.set_ylabel('y (px)');axis.legend(fontsize=7)
            for axis in list(axes.flat)[n:]:axis.axis('off')
        elif kind=='hist':
            numeric(df,[x]);ax.hist(df[x],bins='auto',color=colors[0],edgecolor='white',label='Observaciones');ax.set_ylabel('Frecuencia (N)')
        elif kind=='bar':
            numeric(df,[y]);ax.bar(df[x].astype(str),df[y],color=colors[0],label=y);ax.set_ylim(bottom=0)
        elif kind=='groupbar':
            numeric(df,[y]);pivot=df.pivot_table(index=x,columns=g,values=y,aggfunc='sum');pivot.plot.bar(ax=ax,color=colors)
        elif kind=='paired':
            numeric(df,[y,g]);a=np.arange(len(df));ax.bar(a-.18,df[y],.36,label=y,color=colors[0]);ax.bar(a+.18,df[g],.36,label=g,color=colors[1]);ax.set_xticks(a,df[x].astype(str));ax.set_ylim(bottom=0)
        elif kind in ('scatter','bubble','pareto'):
            numeric(df,[x,y]);sizes=np.full(len(df),70.)
            if kind=='bubble':
                size='map50_95' if y=='gflops' else 'params_m';numeric(df,[size]);sizes=80+300*df[size]/max(df[size].max(),1e-12)
            ax.scatter(df[x],df[y],s=sizes,color=colors[0],label='Modelos; area='+('map50_95' if y=='gflops' else 'params_m') if kind=='bubble' else 'Observaciones')
            if g in df:
                for _,r in df.iterrows():ax.annotate(str(r[g]),(r[x],r[y]),xytext=(5,5),textcoords='offset points',fontsize=9)
            if kind=='pareto':
                best=-float('inf');front=[]
                for _,r in df.sort_values([x,y],ascending=[True,False]).iterrows():
                    if r[y]>best:front.append((r[x],r[y]));best=r[y]
                if front:ax.plot(*zip(*front),color=colors[1],ls='--',label='Frontera Pareto (min x, max y)')
        elif kind in ('box','interval'):
            numeric(df,[y]);groups=list(df.groupby(x,sort=True));labels=[f'{k}\nN={len(v)}' for k,v in groups]
            if kind=='box':ax.boxplot([v[y].to_numpy() for _,v in groups],tick_labels=labels)
            else:
                means=[v[y].mean() for _,v in groups];sd=[v[y].std(ddof=1) if len(v)>1 else 0 for _,v in groups]
                ax.errorbar(np.arange(len(groups)),means,yerr=sd,fmt='o',capsize=4,color=colors[0],label='Media ± DE; DE no definida si N=1');ax.set_xticks(np.arange(len(groups)),labels)
        elif kind=='hp_score_heatmap':
            numeric(df,['value','score']);cells=[]
            for parameter,v in df.groupby('parameter',sort=True):
                count=v.value.nunique()
                if count==1:
                    cells.append({'parameter':parameter,'quantile_bin':'Q1','score_mean':v.score.mean(),'N_trials':len(v),'value_min':v.value.min(),'value_max':v.value.max()})
                else:
                    bins=pd.qcut(v.value,q=min(5,count),duplicates='drop')
                    for i,(interval,group) in enumerate(v.groupby(bins,observed=True)):
                        cells.append({'parameter':parameter,'quantile_bin':'Q'+str(i+1),'score_mean':group.score.mean(),'N_trials':len(group),'value_min':group.value.min(),'value_max':group.value.max()})
            binned=pd.DataFrame(cells)
            pivot=binned.pivot(index='parameter',columns='quantile_bin',values='score_mean')
            im=ax.imshow(np.ma.masked_invalid(pivot.to_numpy()),aspect='auto',cmap='Blues',vmin=0,vmax=1);fig.colorbar(im,ax=ax,label='Media de mAP@0.5:0.95 por bin')
            ax.set_xticks(range(len(pivot.columns)),pivot.columns);ax.set_yticks(range(len(pivot)),pivot.index)
            for i,pname in enumerate(pivot.index):
                for j,qname in enumerate(pivot.columns):
                    row=binned[(binned.parameter==pname)&(binned.quantile_bin==qname)]
                    if len(row):ax.text(j,i,'N='+str(int(row.iloc[0].N_trials)),ha='center',va='center',fontsize=9)
            binned.to_csv(out/(spec['id']+'.bins.csv'),index=False)
        elif kind in ('matrix','heatmap'):
            numeric(df,[y]);pivot=df.pivot_table(index=g,columns=x,values=y,aggfunc='sum' if kind=='matrix' else 'mean')
            if pivot.isna().any().any():raise InsufficientData('Combinaciones no observadas; registrar ceros estructurales explicitamente')
            im=ax.imshow(pivot.to_numpy(),cmap='Blues',aspect='auto');fig.colorbar(im,ax=ax,label=spec['unit']);ax.set_xticks(range(len(pivot.columns)),pivot.columns);ax.set_yticks(range(len(pivot)),pivot.index)
            if kind=='matrix':
                for i in range(len(pivot)):
                    for j in range(len(pivot.columns)):ax.text(j,i,str(pivot.iloc[i,j]),ha='center',va='center')
        elif kind=='parallel':
            numeric(df,[y,'score']);pivot=df.pivot(index=g,columns=x,values=y)
            if pivot.isna().any().any():raise InsufficientData('Hyperparametros incompletos/categoricos sin codificacion declarada')
            ranges=pivot.max()-pivot.min();norm=(pivot-pivot.min())/ranges.replace(0,1)
            scores=df.groupby(g)['score'].agg(['first','nunique'])
            if (scores['nunique']>1).any():raise InsufficientData('Trial con score inconsistente')
            norm['mAP@0.5:0.95']=scores['first']
            cmap=plt.get_cmap('viridis')
            for i,row in norm.iterrows():ax.plot(range(len(row)),row,color=cmap(float(scores.loc[i,'first'])),alpha=.6)
            from matplotlib.cm import ScalarMappable
            fig.colorbar(ScalarMappable(norm=plt.Normalize(0,1),cmap=cmap),ax=ax,label='mAP@0.5:0.95')
            ax.set_xticks(range(len(norm.columns)),norm.columns);ax.set_ylabel('Hiperparametros min/max; mAP en fraccion; ver CSV')
        elif kind=='radar':
            fields=['map50_95','map50','precision','recall','f1'];numeric(df,fields);plt.close(fig);fig,ax=plt.subplots(figsize=(9,7),subplot_kw={'projection':'polar'});angles=np.linspace(0,2*np.pi,len(fields),endpoint=False).tolist();angles+=[angles[0]]
            for i,(_,r) in enumerate(df.iterrows()):v=r[fields].tolist();ax.plot(angles,v+[v[0]],label=str(r[x]),color=colors[i%5])
            ax.set_xticks(angles[:-1],fields);ax.set_ylim(0,1)
        elif kind.startswith('two_panel'):
            score='map50_95' if kind.endswith('det') else 'ap_positive';numeric(df,['epoch',score])
            if 'run_id' not in df:raise InsufficientData('Historial principal requiere run_id para N real por epoca')
            if df.duplicated(['run_id','epoch']).any():raise InsufficientData('Epoca duplicada en el mismo run')
            counts=df.groupby('epoch')['run_id'].nunique()
            counts.rename('N_runs_observados').to_csv(out/(spec['id']+'.N_por_epoca.csv'))
            loss=[c for c in df if (c.startswith('train/') or c.startswith('val/')) and 'loss' in c] if kind.endswith('det') else ['train_loss','val_loss']
            train=[c for c in loss if c.startswith('train')];val=[c for c in loss if c.startswith('val')]
            if not train or not val:raise InsufficientData('Faltan losses autenticas train/validation interna; no usar 20% por epoca')
            numeric(df,loss)
            for i,c in enumerate(loss):
                agg=df.groupby('epoch')[c].agg(['mean','std','count']);ax.plot(agg.index,agg['mean'],label=c,color=colors[i%5],ls='--' if c.startswith('val') else '-')
                if (agg['count']>1).any():ax.fill_between(agg.index,agg['mean']-agg['std'],agg['mean']+agg['std'],alpha=.12,color=colors[i%5])
            agg=df.groupby('epoch')[score].agg(['mean','std','count']);axes[1].plot(agg.index,agg['mean'],color=colors[0],label=score)
            if (agg['count']>1).any():axes[1].fill_between(agg.index,agg['mean']-agg['std'],agg['mean']+agg['std'],color=colors[0],alpha=.15)
            ax.set_title('(a) Loss: componentes originales');ax.set_ylabel('Loss (unidad del backend)');axes[1].set_title('(b) '+score);axes[1].set_ylabel(score+' (fraccion)')
            for axis in axes:axis.set_xlabel('Epoca');axis.legend(fontsize=8);axis.grid(alpha=.2)
        else:
            numeric(df,[x,y]);series=[y]
            if kind=='multiline':series=[c for c in ([y,'map50','f1'] if y=='map50_95' else [y,'accuracy','f1'] if y=='ap_positive' else [y,'recall'] if y in ('precision','f1') else [y,'val_loss']) if c in df]
            for i,c in enumerate(series):
                groups=list(df.groupby(g,sort=True)) if g in df else [('observaciones',df)]
                for j,(name,v) in enumerate(groups):
                    v=v.sort_values(x);ax.plot(v[x],v[c],label=f'{name}: {c}',color=colors[(i+j)%5],ls='--' if i else '-')
                    if kind=='bestline':ax.plot(v[x],v[c].cummax(),label=str(name)+' mejor acumulado',color=colors[(i+j)%5],ls='--')
        if not kind.startswith('two_panel') and kind not in ('images','radar'):
            ax.set_xlabel("Cuantil por hiperparametro (rangos en .bins.csv)" if kind=="hp_score_heatmap" else x or spec['unit'])
            if kind=='hp_score_heatmap':ax.set_ylabel('Hiperparametro')
            elif kind!='hist':ax.set_ylabel((y or 'Frecuencia')+' ('+spec['unit']+')')
            handles,labels=ax.get_legend_handles_labels()
            if handles:ax.legend(fontsize=8)
            ax.grid(axis='y',alpha=.2)
            if len(ax.get_xticks())>8:ax.tick_params(axis='x',labelrotation=35)
        fig.suptitle(spec['title'],fontsize=14)
        n=df['run_id'].nunique() if 'run_id' in df else len(df)
        footer=f"N={n} ({'runs' if 'run_id' in df else 'filas'}); denominador={item['denominator']} | {item['dataset']} / {item['split']} / {item['phase']}\nFuente: {p.name} SHA256 {sha(p)[:16]}; metrica={item['metric']}; config={str(item['configuration'])[:90]}"
        
        if kind.startswith('two_panel'):footer+='; N/epoca='+str(int(counts.min()))+'..'+str(int(counts.max()))+' (ver CSV)'
        fig.text(.02,.02,footer,fontsize=8);fig.tight_layout(rect=(0,.12,1,.93))
        paths=[]
        for fmt in ('png','svg','pdf'):
            dest=out/(spec['id']+'.'+fmt);fig.savefig(dest,dpi=300,bbox_inches='tight');paths.append(dest)
        df.to_csv(out/(spec['id']+'.data.csv'),index=False)
        return paths
    finally:plt.close(fig)
def generate(out,index=None,catalog_only=False,allow_pixels=False,allow_test=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[]
    for spec in catalog():
        row={**spec,'generated_utc':datetime.now(timezone.utc).isoformat(),'source':'','split':'PENDIENTE','metric':spec['y'],'denominator':'PENDIENTE','configuration':'PENDIENTE','N':None,'path':'','sha256':'','status':'PENDIENTE: datos insuficientes','reason':spec['required_artifact']+'; '+spec['required_command']}
        try:
            if spec['kind'] in ('diagram','reference'):
                paths,source=conceptual(spec,out);row.update(status='ESQUEMA_METODOLOGICO' if spec['kind']=='diagram' else 'REFERENCIA_NO_IMPLEMENTADA',source=source,source_sha256=sha(Path(__file__)),split='no aplica (esquema)',phase='metodologia/conceptual',metric='no aplica: sin mediciones',unit='no aplica',configuration='semantica v4; matching tecnico inicial IoU>=0.50 configurable; sin pesos',denominator='no aplica',N='no aplica',reason='No representa resultados experimentales')
            elif catalog_only:raise InsufficientData('Datos no leidos: catalog-only; '+spec['required_artifact'])
            else:
                df,item,p=load_source(spec,index or {},allow_test);paths=render(spec,df,item,p,out,allow_pixels);row.update(status='GENERADA_DESDE_DATOS',source=str(p),split=item['split'],phase=item['phase'],metric=item['metric'],unit=item['unit'],denominator=item['denominator'],configuration=item['configuration'],N=len(df),reason='',source_sha256=sha(p))
            row.update(path=';'.join(str(p) for p in paths),sha256=';'.join(sha(p) for p in paths))
        except (InsufficientData,ImportError,FileNotFoundError,KeyError,ValueError,TypeError) as exc:row['reason']=type(exc).__name__+': '+str(exc)
        rows.append(row)
    (out/'figure_manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (out/'figure_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v for k,v in r.items()} for r in rows)
    return rows
