# PC-03: aprobar e iniciar o reanudar el HPO R50-vd

El usuario ya autorizó continuar con el protocolo original. La aprobación se registra solo si la auditoría y las condiciones críticas pasan. No certifica ausencia de sobreajuste ni inspección visual humana.

El comando descarga tres scripts del commit fijado, comprueba SHA256 y ejecuta el controlador con la autorización explícita. No cambia HEAD, no repite baseline ni inicia robustez/TEST. Reutiliza el mismo comando para recuperar una interrupción propia verificable.

```cmd
cd /d "%USERPROFILE%\Documents\s2_no_borrar\Entrenamiento\Botrytis_Experiment_Framework_git" && git fetch origin fecc8b455aba6b135fdd5538a948cb047d66e825 && ".venv\Scripts\python.exe" -u -c "from pathlib import Path;import subprocess,hashlib,sys;d=Path.home()/'r50/control/fecc8b45';d.mkdir(parents=True,exist_ok=True);files=[('botrytis_r50_hpo_pc03.py', '160ce95cb02e3ce2a4decf4a2698e5646032b05bbfcbb6b0887b0a8bfef806e7'), ('botrytis_r50_auditoria.py', '432881891c91bc6e20becc7fec97d245a657ce5c7e6f1677fb99c224d344149e'), ('botrytis_r50_oficial_pinned.py', 'e62e47fdc4cdbe6648d0830a8efac9701ab04ae82b77757d4a0880ec73381df7')];pairs=[(d/n,subprocess.check_output(['git','show','fecc8b455aba6b135fdd5538a948cb047d66e825:tools/r50_hpo/'+n]),h) for n,h in files];assert all(hashlib.sha256(b).hexdigest()==h for p,b,h in pairs),'Integridad incorrecta';assert all(not p.exists() or p.read_bytes()==b for p,b,h in pairs),'Version distinta; se conserva';[(p.with_name(p.name+'.part').write_bytes(b),p.with_name(p.name+'.part').replace(p)) for p,b,h in pairs if not p.exists()];sys.exit(subprocess.call([sys.executable,'-u',str(d/'botrytis_r50_hpo_pc03.py'),'--aprobar-original-hpo']))"
```

Consulta opcional, después de instalar:

```cmd
cd /d "%USERPROFILE%\Documents\s2_no_borrar\Entrenamiento\Botrytis_Experiment_Framework_git" && ".venv\Scripts\python.exe" -u "%USERPROFILE%\r50\control\fecc8b45\botrytis_r50_auditoria.py" status
```

Progreso esperado: `APROBACION_HPO_REGISTRADA`, `INICIANDO_HPO_RTDETR`, trial/configuración, fold y época. Éxito: `HPO_RTDETR_TERMINADO`; la cola termina normalmente en `PENDING_REVIEW`, sin validación definitiva.

Resultados: `%USERPROFILE%\r50\r\rtdetr_r50\hpo`. Trazabilidad y resumen: `%USERPROFILE%\r50\hpo_control\latest.json`.
