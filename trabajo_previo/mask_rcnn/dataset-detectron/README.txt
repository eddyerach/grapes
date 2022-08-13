### README 0.1 ##############################

== Dibujar etiquetas (json2img.py) ==================================================
Como usar:
1-Exportar las etiquetas desde VIA como json
2-Abrir json2img.py
3-Especificar el nombre del json en los parametros (asegurarse que este en la misma carpeta)
4-Correr json2img.py
5-Las imagenes estan en la carpeta annotations_img

== Crear dataset detectron2 y dibujar mask con bounding box (json2mask.py) ==========
Como usar:
1-Exportar las etiquetas desde VIA como json
2-Abrir json2mask.py
3-Especificar el nombre del json en los parametros (asegurarse que este en la misma carpeta)
4-Especificar el numero de imagenes con etiqueta lista en los parametros
5-Correr json2mask.py
6-El dataset es dataset.json, las imagenes estan en la carpeta annotations_img_2

NOTA: Asegurarse de que el json de VIA tenga la direccion a las imagenes correctamente.
Para arreglarlas simplemente abrir el json con un editor de texto y reemplazar el path antiguo por el nuevo
