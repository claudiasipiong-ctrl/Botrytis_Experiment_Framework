# Comandos PC-03: auditoría y consulta de RT-DETR-R50-vd

Código fijado: `1b14600debd2f2f01e35bcf33848723b7dd8b462`. No hacer checkout/pull/reset. No cambia baseline ni estados de cola. No entrena.

## Instalar y auditar desde cualquier carpeta de CMD

```cmd
cd /d "%USERPROFILE%\Documents\s2_no_borrar\Entrenamiento\Botrytis_Experiment_Framework_git" && git fetch origin 1b14600debd2f2f01e35bcf33848723b7dd8b462 && ".venv\Scripts\python.exe" -u -c "from pathlib import Path;import subprocess,hashlib,sys;d=Path.home()/'r50/control/1b14600d';d.mkdir(parents=True,exist_ok=True);files=[('botrytis_r50_auditoria.py', '432881891c91bc6e20becc7fec97d245a657ce5c7e6f1677fb99c224d344149e'), ('botrytis_r50_oficial_pinned.py', 'e62e47fdc4cdbe6648d0830a8efac9701ab04ae82b77757d4a0880ec73381df7')];pairs=[(d/n,subprocess.check_output(['git','show','1b14600debd2f2f01e35bcf33848723b7dd8b462:tools/r50_audit/'+n]),h) for n,h in files];assert all(hashlib.sha256(b).hexdigest()==h for p,b,h in pairs),'Integridad incorrecta';assert all(not p.exists() or p.read_bytes()==b for p,b,h in pairs),'Version distinta; se conserva';[(p.with_name(p.name+'.part').write_bytes(b),p.with_name(p.name+'.part').replace(p)) for p,b,h in pairs if not p.exists()];sys.exit(subprocess.call([sys.executable,'-u',str(d/'botrytis_r50_auditoria.py'),'audit']))"
```

## Consultar después de instalar

```cmd
cd /d "%USERPROFILE%\Documents\s2_no_borrar\Entrenamiento\Botrytis_Experiment_Framework_git" && ".venv\Scripts\python.exe" -u "%USERPROFILE%\r50\control\1b14600d\botrytis_r50_auditoria.py" status
```

Resultados de auditoría: `%USERPROFILE%\r50\audit\<id>\AUDITORIA.html`, `audit.json` y curvas por fold. `latest.json` indica el informe reciente.

Si la auditoría emite `B_APTO_CON_OBSERVACIONES`, revisar el informe y responder `APROBAR` para recibir el ejecutor HPO. Si emite C/D, conserva todo y enumera errores/evidencias faltantes. Conforme a la instrucción 7 del usuario, el comando de aprobación/inicio HPO se entrega después de su autorización explícita; esta entrega no lo ejecuta ni lo simula.
