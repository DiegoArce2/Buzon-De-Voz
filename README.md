# Buzon-De-Voz
Proyecto encargado por la empresa Xentric para detectar buzones de voz en su bot telefonico y asi ahorrar recursos.

# Bibliotecas necesarias:
-Flask
-Os
-Glob
-Librossa
-Numpy
-Matplotlib
-Soundfile
-Keras
-Pandas
-Openpyxl
-Zipfile
-Json
-Tensorflow

# Archivos importantes:
 -Archivo "ApiCategorizacion.py": En este archivo se ejecuta todo el trabajo incluyendo el modelo y el api.
 -Archivo "Log.xlsx": En este archivo se guarda la informacion de:
  -Errores
  -Entrenamientos
  -Categorizaciones.

# Directorios:
-Directorio data: Aqui se almacenan las categorias y los archivos de entrenamiento de los modelos.
-Directorio https: Se almacenan los certificados para utilizar el "Protocolo seguro de transferencia de hipertexto".
-Directorio models: Se almacenan los modelos ya entrenados.
-Directorio npys: Se almacenan los datos extraidos de los audios de cada modelo.
-Directorio predict: Se almacena temporalmente el audio a categorizar.
