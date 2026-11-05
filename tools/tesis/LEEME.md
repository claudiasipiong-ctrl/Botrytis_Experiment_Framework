# Exportador de evidencias de tesis Botrytis

Ejecutar en cada PC con su .venv cuando no tenga otros procesos Python activos. No requiere un arbol manual de carpetas. Detecta PC por usuario y lee rondas.json/configuraciones. Recorre las rutas cortas conocidas y las declaradas, incluidos antecedentes y auditorias. No recorre todo el perfil personal.

Genera un ZIP unico por computadora bajo Downloads/Botrytis_Tesis/PC-XX_fecha_id, con LEEME.md y huella .zip.sha256.txt. Incluye codigo, configuraciones y protocolo del repositorio, entornos/QA, particiones, tablas, estudios y colas SQLite, historiales, predicciones, figuras, evidencias visuales y pesos final.pt disponibles. Documentos de tesis fuera del repositorio se agregan manualmente a Drive. No inventa mediciones ni ejecuta analisis.

PC-01 agrega las imagenes y etiquetas originales de TRAIN/VAL para compartirlas una sola vez. Las tablas derivadas se incluyen en las tres PCs; el dataset completo y TEST no se exportan. OFF/_off se excluyen. CSV con filas TEST y registros JSON de TEST se filtran solo en la copia exportada y se registra la transformacion. split_manifest.csv se limita a ID/split. Los metadatos publicos pueden conservar IDs/conteos de TEST, sin imagenes/anotaciones/predicciones. No se ejecuta inferencia ni se abre TEST para evaluarlo.

Cada archivo tiene ruta original, ruta ZIP y SHA256 de origen/exportacion en 00_INDICES/INVENTARIO.csv. Los SQLite se exportan con backup consistente, sin WAL sueltos, y llevan hash del snapshot; no hay instantanea transaccional global de toda la computadora. Se requiere inactividad de Python para evitar interferir con los controladores de entrenamiento. Archivos cambiantes o ilegibles quedan registrados en ERRORES y el paquete se etiqueta INCOMPLETO.

Los checkpoints de optimizer/rolling/best de los trials y los pesos preentrenados grandes no se copian ni se borran: sus rutas/tamanos se inventarian en EXCLUIDOS. Los pesos finales si se incluyen. El paquete permite rehacer graficos a partir de predicciones e historiales; no es un backup completo para reanudar cada trial ni garantiza cualquier grafico futuro.

FALTANTES enumera salidas de etapas aun no generadas. No significa que hayan fallado ni autoriza reentrenarlas. Rutas largas dentro del ZIP se abrevian y mapean en requested_archive; extraiga en una carpeta corta como C:\bt_tesis\PC01. Los recibos/indices originales conservan sus hashes de origen: use el inventario del ZIP para verificar copias filtradas.

Drive sugerido: Botrytis_Tesis/01_Paquetes_originales/PC-01, PC-02, PC-03; 02_Datos_para_analisis; 03_Graficos_editables; 04_Tablas_tesis; 05_Texto_experimentacion_resultados. Mantenga ZIP originales por fecha y analice copias extraidas en carpetas por PC/ronda; no mezcle CSV homonimos. No sube archivos automaticamente a Drive.

19 pruebas portables aprobadas; sin entrenamientos ni ejecucion Windows real en la nube.
