# Inventario documental final, tres PCs

Ejecutar en cada PC con Python de su .venv. Usuario i21315 -> PC-01; i21314 -> PC-02; i21313 -> PC-03.
No ejecuta entrenamiento/inferencia, no consulta SQLite ni lee contenidos de resultados/pesos.
Lee nombres/tamanos/fechas en carpetas conocidas de Botrytis y Git commit/status del repositorio.
Genera Downloads/Inventario_Tesis_PC-XX_<fecha>_<id>.zip con ARBOL.txt, ARCHIVOS.csv y RESUMEN.json.
El ZIP solo contiene inventarios; no sustituye el posterior paquete de evidencias.

Incluye bd para los dos TEST finales, bf y rondas, auditorias, logs, r50 parcial y codigo/config/docs
locales. Registra pero no desciende a crops/images/labels/visualizations: los ejemplos visuales se
seleccionaran al empaquetar. Si estan bajo plots/figures, los archivos de figuras si se enumeran.
Omite caches, entornos virtuales, enlaces/junctions y archivos de credenciales conocidos.
No recorre AppData ni el perfil completo. Carpetas ausentes no se crean ni se consideran errores.
Las fechas son UTC y los modelos asignados son referencias del plan, no estados inferidos.

Validacion: 3 pruebas CPU Linux: cobertura de TEST y RT-DETR parcial, registro de imagenes masivas,
conservacion de fuentes y exclusion de enlaces/credenciales. Windows no ejecutado desde aqui.
