# Buzon-De-Voz
Proyecto encargado por la empresa Xentric para detectar buzones de voz en su bot telefonico y asi ahorrar recursos.

# Bibliotecas necesarias:
-Flask<br>
-Os<br>
-Glob<br>
-Librossa<br>
-Numpy<br>
-Matplotlib<br>
-Soundfile<br>
-Keras<br>
-Pandas<br>
-Openpyxl<br>
-Zipfile<br>
-Json<br>
-Tensorflow<br>

# Archivos importantes:
 -Archivo "ApiCategorizacion.py": En este archivo se ejecuta todo el trabajo incluyendo el modelo y el api.<br>
 -Archivo "Log.xlsx": En este archivo se guarda la informacion de:<br>
  -Errores<br>
  -Entrenamientos<br>
  -Categorizaciones.<br>

# Directorios:
-Directorio data: Aqui se almacenan las categorias y los archivos de entrenamiento de los modelos.<br>
-Directorio https: Se almacenan los certificados para utilizar el "Protocolo seguro de transferencia de hipertexto".<br>
-Directorio models: Se almacenan los modelos ya entrenados.<br>
-Directorio npys: Se almacenan los datos extraidos de los audios de cada modelo.<br>
-Directorio predict: Se almacena temporalmente el audio a categorizar.<br>
