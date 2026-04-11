import numpy as np

documents = [
    "ARTÍCULO 30: La matrícula es el acto mediante el cual la Institución reconoce formalmente a una persona como estudiante.",
    "ARTÍCULO 33: La matrícula causa el pago de derechos pecuniarios a la Institución, los cuales serán determinados por el Consejo Superior para cada periodo académico, de acuerdo con las normas legales vigentes.",
    "ARTÍCULO 34: La asignación académica máxima semanal para todos los estudiantes será el número de créditos establecidos para cada periodo académico en el plan de estudios del programa al cual se matriculó.",
    "ARTÍCULO 37: La persona que reingresa debe asumir las reformas curriculares o de modalidades que hubieren ocurrido durante su ausencia, así como los incrementos en el valor de derechos pecuniarios.",
    "ARTÍCULO 42: La asistencia de los estudiantes es obligatoria, toda vez que el programa académico al que está matriculado disponga de espacios formativos que impliquen presencialidad o mediación tecnológica de esta.",
    "ARTÍCULO 47: El Consejo Académico fija, en cada periodo lectivo, las fechas de registro y cancelación de asignaturas.",
    "ARTÍCULO 57: Las calificaciones de cada asignatura se expresarán en números enteros en una escala comprendida entre cero (0) y cincuenta (50) puntos.",
    "ARTÍCULO 1: La Institución podrá ofrecer programas académicos en todos los niveles formales de la educación superior establecidos por la ley: técnico profesional, tecnológico, profesional universitario, especialización, maestría y doctorado.",
    "ARTÍCULO 2: La Institución podrá desarrollar programas de educación informal, los cuales no conducirán a la obtención de títulos profesionales ni a certificaciones de aptitud ocupacional.",
    "ARTÍCULO 3: Los títulos otorgados por la Institución se ajustan a lo previsto en la Ley.",
    "ARTÍCULO 5: De conformidad con lo que establezca la ley, la unidad de trabajo académico en la Institución es el crédito académico.",
    "ARTÍCULO 6: El número de créditos de un programa académico y de los espacios formativos que lo integren será el definido por el Consejo Superior en consideración a las recomendaciones hechas por el Consejo Académico.",
    "ARTÍCULO 7: El número de créditos académicos de una actividad en el plan de estudios será el resultado de dividir entre 48 el total de horas que el estudiante debe emplear para cumplir con los resultados de aprendizaje esperados.",
    "ARTÍCULO 8: En el ejercicio de su autonomía y conforme a las características del programa, la Institución distinguirá entre créditos académicos obligatorios y electivos.",
    "ARTÍCULO 10: Cada periodo académico tendrá una duración definida en semanas, conforme lo reglamente el Consejo Académico, considerando las características de los planes de estudio de los programas académicos, las Resoluciones de Registro Calificado y la normativa nacional vigente.",
    "ARTÍCULO 14: La Institución se reserva el derecho de seleccionar a los estudiantes y define, para cada programa, los criterios de admisión basados en el perfil de ingreso aprobado por el Ministerio de Educación Nacional. ",
    "ARTÍCULO 17: Una vez culminado el proceso de admisión, la Institución informará al aspirante, los resultados correspondientes."
]

# Termino a buscar
termino_buscado = "créditos"

# Total de documentos
N = len(documents)

# Calcular IDF
df = sum(1 for doc in documents if termino_buscado in doc)
idf = np.log(N / (df))

print(f"IDF para el termino '{termino_buscado}': {idf:.4f}")

# Calcular TF-IDF para cada documento
scores = []
for i, doc in enumerate(documents):
    tf = doc.count(termino_buscado) / len(doc.split())
    tf_idf = tf * idf
    scores.append((i+1, tf_idf, doc))

scores.sort(key=lambda x: x[1], reverse=True)  # Ordenar de mayor a menor

# Mostrar resultados
for i, (doc_id, tf_idf, doc) in enumerate(scores,1):
    print(f" {i} - Documento: {doc}: TF-IDF = {tf_idf:.4f}")