#!/usr/bin/env python
# coding: utf-8

# In[29]:


import csv
import pandas as pd
archivo_csv = "C://Users//ASUS//Documents//SOFIA//IA//DATA//Linkedin_TI_2k_SV.csv"

df = pd.read_csv('Linkedin_TI12k.csv')


# In[30]:


df.head()


# In[31]:


df.info()


# #### Observando los datos faltantes

# In[32]:


# Ver los valores faltantes por columna
missing_values = df.isnull().sum()
print("Valores faltantes por columna:")
print(missing_values)

# Ver los valores faltantes en todo el DataFrame
print("\nValores faltantes en el DataFrame:")
print(df.isnull())


# In[5]:


# Ver las columnas que contienen "No description available"
columns_with_description = df.isin(['No description available']).any()
print("Columnas con 'No description available':")
print(columns_with_description)

# Contar cuántas veces aparece "No description available" en cada columna
count_description = df.isin(['No description available']).sum()
print("\nCantidad de veces que aparece 'No description available' por columna:")
print(count_description)


# In[33]:


len(df)


# In[34]:


import pandas as pd

# Filtrar y eliminar las filas que contienen 'No description available'
df = df[df['Cuerpo'] != 'No description available']

# Mostrar el DataFrame actualizado
print(df.head())  


# In[35]:


print(len(df))


# In[36]:


print(df['Cuerpo'].iloc[0])


# In[37]:


###Separando texto junto

import re

text = """Industria SaludUbicación Ciudad de Panamá, PanamáDescripción de la empresaEmpresa transnacional del ramo de los seguros dedicada ayudar a las personas a mejorar su salud, bienestar y seguridad gracias a una amplia gama de planes y servicios.PropósitoSupervisar el desarrollo y la difusión de tecnología para clientes externos, proveedores y otros clientes con el fin de aumentar los beneficios y cumplir con los objetivos de la empresa. Entiende a corto y largo plazo las necesidades de su empresa, maneja y dirige su equipo, y al mismo tiempo canaliza el flujo de trabajo necesario para terminar los proyectos a tiempo.ObjetivoFormular una visión clara y concisa de cómo se utilizará la tecnología dentro de la empresa, y conseguir que la implantación de la solución tecnológica sea un éxito en la consecución de objetivos, así como en el empleo de los recursos y esfuerzos previstos.RequisitosMás de 15 años de experiencia en la gestión del área de TI para empresas del área de Salud.Licenciatura en un campo relacionadoMás de 5 años de trabajo técnico práctico como ingeniero de sistemasMás de 5 años gestionando un equipo de 10 o más personas.Capacidad demostrada para comunicar eficazmente cuestiones técnicas complejas a partes interesadas no técnicasExcepcionales habilidades verbales, escritas y organizativas.Capacidad para gestionar y priorizar solicitudes y proyectos simultáneos"""

# Utilizamos una expresión regular para encontrar palabras que están juntas
pattern = r'([a-záéíóúñü])([A-ZÁÉÍÓÚÑÜ])'

# Reemplazar las palabras juntas encontradas por palabras separadas con un espacio
separated_text = re.sub(pattern, r'\1 \2', text)

print(separated_text)


# #### SEPARANDO TEXTO JUNTO DE LA COLUMNA CUERPO (DESCRIPCIONES) 

# In[38]:


import pandas as pd
import re

# Esta es la función mejorada para separar palabras juntas en una cadena de texto
def separate_words(text):
    # Expresión regular mejorada para separar palabras juntas
    pattern = r'(?<=[a-záéíóúñü])(?=[A-ZÁÉÍÓÚÑÜ])|(?<=[A-ZÁÉÍÓÚÑÜ])(?=[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü])'

    return re.sub(pattern, ' ', text)

# Aplicar la función separate_words a todas las filas de la columna 'Cuerpo'
df['Cuerpo'] = df['Cuerpo'].apply(separate_words)

# Mostrar el DataFrame actualizado
print(df.head())  # Esto imprimirá las primeras filas del DataFrame con la columna 'Cuerpo' actualizada


# In[40]:


df["Cuerpo"].iloc[1]


# In[41]:


df.head()


# #### Una vez lista la separación de palabras, traduciremos al ingles todas las descripciones

# In[42]:


import pandas as pd
from mtranslate import translate

# Esta función traduce un texto a un idioma específico
def translate_to_english(text):
    try:
        # Traducir la descripción al inglés
        translated_description = translate(text, 'en')
        return translated_description
    except Exception as e:
        print(f"Error al traducir: {e}")
        return None

df['Cuerpo_en'] = df['Cuerpo'].apply(translate_to_english)

# Mostrar el DataFrame con la nueva columna 'Cuerpo_en'
print(df[['Cuerpo', 'Cuerpo_en']].head())


# In[43]:


df["Cuerpo_en"].iloc[1]


# In[44]:


import pandas as pd
import os

folder_name = "C://Users//ASUS//Documents//SOFIA//IA//DATA"

# Verificar si la carpeta existe, si no, crearla
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Ruta del archivo CSV en la carpeta 'data'
file_path = os.path.join(folder_name, 'data_english.csv')

# Guardar el DataFrame como un archivo CSV en la carpeta 'data'
df.to_csv(file_path, index=False)

print(f"El archivo se ha guardado en: {file_path}")


# In[1]:


import csv
import pandas as pd
archivo_csv = "C://Users//ASUS//Documents//SOFIA//IA//DATA//data_english.csv"

df = pd.read_csv('data_english.csv')


# In[2]:


df.head()


# ### PREPROCESAMIENTO 

# #### Eliminando palabras vacias (Stop Words)

# In[3]:


import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

df['Cuerpo_en'] = df['Cuerpo_en'].fillna('')  # Rellenar los valores nulos con una cadena vacía

stop_words = set(stopwords.words('english'))

# Función para eliminar las stopwords de un texto
def remove_stopwords(text):
    # Tokenizar el texto en palabras
    words = word_tokenize(text)
    
    filtered_words = [word for word in words if word.lower() not in stop_words]
    
    return ' '.join(filtered_words)

df['Cuerpo_en'] = df['Cuerpo_en'].apply(remove_stopwords)

# Mostrar el DataFrame con la columna 'Cuerpo_en' actualizada
print(df[['Cuerpo_en']].head())


# In[4]:


print(df["Cuerpo"].iloc[0])
print("-------------------------------------")
print(df["Cuerpo_en"].iloc[0])


# ### Obteniendo las Skills de cada puesto de trabajo

# In[ ]:


import pandas as pd
import numpy as np
import tensorflow as tf

# Cargar el dataset
dataset = pd.read_csv("data_english.csv", dtype=str)

# Crear el vocabulario
words = set()
for description in dataset["Cuerpo_en"]:
    words.update(description.split())

# Convertir palabras a enteros
word_to_index = {word: i for i, word in enumerate(words)}

# Crear el modelo de lenguaje
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=len(word_to_index), output_dim=128),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(len(word_to_index))
])

# Cargar el modelo
model.load_weights("model.h5")

# Procesar las descripciones de trabajo
skills = []
for description in dataset["Cuerpo_en"]:
    prediction = model.predict(description)
    skills.append(prediction)

# Agregar la columna de skills al dataset
dataset["Skills"] = skills

# Guardar el dataset
dataset.to_csv("data_with_skills.csv")


# In[59]:


import pandas as pd

# Cargar el dataset
dataset = pd.read_csv("data_english.csv")

# Encontrar los datos faltantes
is_null = dataset["Cuerpo_en"].isnull()

# Imprimir los datos faltantes
print(is_null)


# In[62]:


import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def extract_skills(description):
    # Tokenizar el texto en palabras
    words = word_tokenize(description.lower())  # Convertir a minúsculas
    
    # Filtrar palabras que no son stopwords ni signos de puntuación
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    
    # Calcular la frecuencia de las palabras
    word_freq = Counter(filtered_words)
    
    # Obtener las palabras más comunes como habilidades
    # Puedes ajustar el número de palabras clave a extraer cambiando el valor de 'num_keywords'
    num_keywords = 100  # Cambiar este valor según tus necesidades
    skills = [word for word, _ in word_freq.most_common(num_keywords)]
    
    return skills

# Aplicar la función extract_skills a la columna 'Cuerpo' para obtener las habilidades de cada descripción de trabajo
df['Skills'] = df['Cuerpo_en'].apply(extract_skills)

# Mostrar las habilidades encontradas para cada puesto de trabajo
print(df[['Cuerpo_en', 'Skills']].head())


# In[63]:


df.head()


# #### Obteniendo los requerimientos de cada puesto de trabajo

# In[70]:


import re

text = df["Cuerpo_en"].iloc[1]

def extract_requirements(text):
    # Expresiones regulares para encontrar frases que indiquen requisitos
    regex_experience = r'\b(\d+\s*(?:years|year))\b'  # Buscar patrón de años de experiencia
    regex_education = r'(?:Academic training|Specialization)\s+(.*?)\.'  # Buscar formación académica y especialización

    # Buscar coincidencias de patrones en el texto
    experience_matches = re.findall(regex_experience, text)
    education_matches = re.findall(regex_education, text, flags=re.IGNORECASE)

    # Combinar resultados de experiencia y educación
    requirements = {
        'Experience': experience_matches,
        'Education': education_matches
    }

    return requirements

# Aplicar la función extract_requirements al texto para obtener los requisitos
requirements = extract_requirements(text)

# Mostrar los requisitos encontrados
print("Requisitos de experiencia:", requirements['Experience'])
print("Requisitos de educación:", requirements['Education'])


# In[85]:


import pandas as pd
import re

# Función para extraer los requisitos de una descripción de trabajo en inglés
def extract_requirements(description):
    # Expresiones regulares para encontrar frases que indiquen requisitos
    regex_experience = r'experience\s+([\w\s,]+)'
    regex_skills = r'skills\s+([\w\s,]+)'

    # Buscar coincidencias de patrones en la descripción (ignorando mayúsculas y minúsculas)
    experience_matches = re.findall(regex_experience, description.lower())
    skills_matches = re.findall(regex_skills, description.lower())

    # Combinar resultados de experiencia y habilidades
    requirements = {
        'Experience': experience_matches,
        'Skills': skills_matches
    }

    return requirements

# Aplicar la función extract_requirements a la columna 'Cuerpo_en' para obtener los requisitos de cada puesto de trabajo
df['Requirements'] = df['Cuerpo_en'].apply(extract_requirements)

# Mostrar los requisitos encontrados para cada puesto de trabajo
print(df[['Cuerpo_en', 'Requirements']].head())


# In[89]:


df["Cuerpo"].iloc[2]


# In[90]:


df["Requirements"].iloc[2]


# In[122]:


import matplotlib.pyplot as plt
import pandas as pd

fig=plt.figure(figsize=(10, 5), dpi= 80, facecolor='w', edgecolor='k')
df.Titulo.hist()


# In[124]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

# Suponiendo que 'df' es tu DataFrame que contiene las columnas 'Titulo' y 'Cuerpo_en'

# Agrupar por título y concatenar las descripciones
grouped = df.groupby('Titulo')['Cuerpo_en'].apply(lambda x: ' '.join(x))

# Generar la nube de palabras por cada título
for title, descriptions in grouped.items():
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(descriptions)
    
    # Mostrar la nube de palabras
    plt.figure(figsize=(10, 5), dpi=80, facecolor='w', edgecolor='k')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f"Word Cloud for '{title}'")
    plt.axis('off')
    plt.show()


# In[127]:


import pandas as pd

# Suponiendo que 'df' es tu DataFrame que contiene la columna 'Cuerpo_en'

# Convertir el texto a minúsculas y buscar las filas que contienen 'requirement' o 'requirements'
count_requirements = df['Cuerpo_en'].str.lower().str.contains('requirement|requirements').sum()

print(f"La palabra 'requirement' o 'requirements' se encuentra en {count_requirements} filas.")


# In[128]:


import pandas as pd

# Suponiendo que 'df' es tu DataFrame que contiene la columna 'Cuerpo_en'

# Contar las filas que no contienen 'requirement' o 'requirements'
count_not_requirements = (~df['Cuerpo_en'].str.lower().str.contains('requirement|requirements')).sum()

print(f"La palabra 'requirement' o 'requirements' no se encuentra en {count_not_requirements} filas.")


# In[142]:


df["Cuerpo"].iloc[6044]


# In[147]:


import nltk
from nltk.tokenize import word_tokenize
import pandas as pd

# Tokenizar las descripciones de trabajo
df['Tokens'] = df['Cuerpo_en'].apply(word_tokenize)

# Definir palabras clave que indiquen requisitos
required_keywords = ['skills', 'experience', 'education', 'requirements']

# Función para extraer los requisitos
def extract_requirements(tokens):
    requirements = []
    for i, token in enumerate(tokens):
        if token.lower() in required_keywords:
            req = ' '.join(tokens[i+1:])  # Extraer los tokens después de la palabra clave
            requirements.append(req)
    return requirements

# Aplicar la función de extracción de requisitos a cada descripción de trabajo
df['Requisitos_2'] = df['Tokens'].apply(extract_requirements)

# Mostrar los requisitos encontrados
print(df[['Cuerpo_en', 'Requisitos_2']])


# In[153]:


df["Requisitos_2"].iloc[3]


# In[154]:


import spacy
import pandas as pd

# Cargar el modelo de spacy
nlp = spacy.load("en_core_web_sm")

# Suponiendo que 'df' es tu DataFrame con la columna 'Cuerpo_en' que contiene descripciones de trabajo

# Función para extraer los requerimientos usando spacy
def extract_requirements(text):
    doc = nlp(text)
    requirements = []
    for ent in doc.ents:
        if ent.label_ == 'SKILL' or ent.label_ == 'EDUCATION':  # Filtrar por habilidades o educación
            requirements.append(ent.text)
    return list(set(requirements))  # Retornar una lista única de requerimientos

# Aplicar la función a la columna 'Cuerpo_en' para extraer los requerimientos
df['Requisitos'] = df['Cuerpo_en'].apply(extract_requirements)

# Mostrar los requerimientos encontrados
print(df[['Cuerpo_en', 'Requisitos']])


# In[ ]:


df["Requisitos"].iloc[0]


# In[161]:


import re

# Función para extraer requerimientos de un texto
def extract_requirements(text):
    # Patrón de expresión regular para buscar requerimientos
    pattern = r'\b(?:requirements|skills needed|requirements include|key qualifications|qualifications)\b[\w\s:,;-]+'
    
    # Buscar coincidencias con el patrón en el texto y devolver los resultados
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    return matches

# Aplicar la función a la columna "Cuerpo_en" del DataFrame
df['Requerimientos_3'] = df['Cuerpo_en'].apply(extract_requirements)


# In[162]:


df.head()


# In[163]:


df["Requerimientos_3"].iloc[0]


# In[155]:


df.head()


# ### MODELO

# ### Aprendizaje automático supervisado

# ##### El siguiente paso es crear un modelo de lenguaje. Este modelo se entrena en un conjunto de datos de texto, y aprende a identificar patrones en el texto.
# ##### Dividimos en conjuntos de entrenamiento y prueba, y crea el modelo de lenguaje. El modelo de lenguaje consta de tres capas:

# In[ ]:





# In[ ]:





# ### NLP para identificar palabras y frases claves en las descripciones de trabajo para identificar habilidades (Skills) 

# In[5]:


# imports
import spacy
import en_core_web_lg
from spacy.matcher import PhraseMatcher

# import skill extractor
from skillNer.skill_extractor_class import SkillExtractor
from skillNer.general_params import SKILL_DB


# init params of skill extractor
nlp = en_core_web_lg.load()
# init skill extractor
skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)


# In[30]:


get_ipython().system('python -m spacy download es_core_news_lg')


# In[8]:


##### extract skills from job_description
job_description = df['Cuerpo_en'].iloc[1] #4

annotations = skill_extractor.annotate(job_description)


# In[9]:


# inspect annotations
skill_extractor.describe(annotations)


# In[181]:


# Función para procesar los datos en 'Cuerpo_en' y extraer habilidades
def process_and_extract_skills(description):
    annotations = skill_extractor.annotate(description)  # Anotar la descripción
    return extract_skills(annotations)  # Extraer habilidades

# Aplicar la función a cada fila de 'Cuerpo_en' y crear una nueva columna 'Skills'
df['Skills'] = df['Cuerpo_en'].apply(process_and_extract_skills)

# Mostrar las primeras filas para verificar los resultados
print(df[['Cuerpo_en', 'Skills']].head())


# In[183]:


df['Skills']


# In[185]:


df.head()


# In[1]:


####Guardando los datos con las skills en un DataFrame
import os
import pandas as pd

# Suponiendo que tienes un DataFrame llamado 'df' que quieres guardar
# Ruta de la carpeta donde deseas guardar el archivo
carpeta_destino = 'C://Users//ASUS//Documents//SOFIA//IA//DATA'  # Reemplaza con tu ruta deseada

# Si la carpeta no existe, se puede crear
if not os.path.exists(carpeta_destino):
    os.makedirs(carpeta_destino)

# Ruta completa al archivo CSV en la carpeta
ruta_archivo = os.path.join(carpeta_destino, 'data_final01.csv')

# Guardar el DataFrame en formato CSV en la carpeta especificada
df.to_csv(ruta_archivo, index=False)  # index=False para no guardar el índice del DataFrame


# In[204]:


from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Cargar el modelo preentrenado BERT y el tokenizador
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Supongamos que tienes un DataFrame llamado 'df' con los campos 'Título', 'Descripción' y 'Skills'

# Supongamos que tienes habilidades de entrada en formato de texto
habilidades_entrada = "Python Machine Learning Data Analysis"

# Codificar las habilidades de entrada con BERT
inputs = tokenizer(habilidades_entrada, return_tensors='pt', padding=True, truncation=True)
with torch.no_grad():
    outputs = model(**inputs)

# Obtener la representación vectorial de las habilidades de entrada
embedding_entrada = torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()

# Supongamos que 'Skills' en el DataFrame es una columna que contiene las habilidades requeridas de cada empleo
skills_dataset = df['Skills'].tolist()

# Verificar si la lista de habilidades no está vacía y contiene textos válidos
if len(skills_dataset) > 0 and all(isinstance(skill, str) for skill in skills_dataset):
    # Codificar las habilidades del dataset con BERT utilizando un bucle
    embeddings_dataset = []
    for habilidad in skills_dataset:
        encoding = tokenizer(habilidad, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**encoding)
        embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()
        embeddings_dataset.append(embedding)

    embeddings_dataset = torch.tensor(embeddings_dataset)

    # Calcular la similitud de coseno entre las habilidades de entrada y las del dataset
    similaridades = cosine_similarity([embedding_entrada], embeddings_dataset)

    # Obtener los empleos más similares
    empleos_similares = df.loc[similaridades.argsort()[0][::-1][:5]]['Título']
    print("Empleos recomendados:")
    print(empleos_similares)
else:
    print("La lista de habilidades está vacía o contiene datos no válidos")


# In[235]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Supongamos que tienes las habilidades de entrada en formato de texto
habilidades_entrada = "project management"

# Supongamos que 'skills' en el DataFrame contiene las habilidades requeridas de cada empleo como listas
skills_dataset = df['Skills'].apply(lambda skills_list: ' '.join(skills_list)).tolist()

# Unir las habilidades de entrada con las habilidades del dataset
skills_dataset.append(habilidades_entrada)

# Inicializar el vectorizador TF-IDF
vectorizer = TfidfVectorizer()

# Calcular la matriz TF-IDF
tfidf_matrix = vectorizer.fit_transform(skills_dataset)

# Calcular la similitud de coseno entre las habilidades de entrada y las del dataset
similaridades = cosine_similarity(tfidf_matrix[:-1], tfidf_matrix[-1])

# Obtener los índices de los empleos más similares
indices_similares = similaridades.argsort(axis=0)[:-6:-1]  # Obtener los 5 más similares, excluyendo la habilidad de entrada

# Imprimir los títulos de los empleos más similares
empleos_similares = df.iloc[indices_similares.flatten()]['Titulo']
print("Empleos recomendados:")
print(empleos_similares)


# In[55]:


import csv
import pandas as pd
archivo_csv = "C://Users//ASUS//Documents//SOFIA//IA//DATA"

df = pd.read_csv('data_final01.csv')


# In[6]:


get_ipython().system('pip install tensorflow-hub')


# In[3]:


#############OTRO MODELO MÁS 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
from tabulate import tabulate
# Supongamos que tienes un DataFrame 'df' con columnas 'Título' y 'Skills'

# Supongamos que tus habilidades son 'habilidades_entrada'
habilidades_entrada = "project management service management pmi"

# Inicializar el vectorizador TF-IDF
tfidf = TfidfVectorizer(stop_words='english')

# Concatenar las habilidades en una sola cadena
df['Skills_text'] = df['Skills'].apply(lambda x: ' '.join(x))

# Agregar las habilidades de entrada a una lista y luego al DataFrame
nuevo_empleo = pd.DataFrame({'Titulo': ['Habilidades de Entrada'], 'Skills_text': [habilidades_entrada]})
df_combined = pd.concat([df, nuevo_empleo], ignore_index=True)

# Aplicar TF-IDF al texto de habilidades
tfidf_matrix = tfidf.fit_transform(df_combined['Skills_text'])

# Calcular similitud del coseno entre las habilidades de entrada y los empleos
cosine_sim = linear_kernel(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

# Obtener los índices de los empleos más similares
related_jobs_indices = cosine_sim.argsort()[:-6:-1]  

# Imprimir los títulos de los empleos recomendados
print("Empleos recomendados:")
for i, index in enumerate(related_jobs_indices, start=1):
    title = df.iloc[index]['Titulo']
    link = df.iloc[index]['Link']
    skills = df.iloc[index]['Skills']
    
    print(f"{i}. Titulo: {title}")
    print(f"   Link: {link}")
    print(f"   Skills: {skills}")
    print("\n")


# In[66]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

# Supongamos que tienes un DataFrame 'df' con columnas 'Título' y 'Skills'
# También se supone que ya tienes definida 'habilidades_entrada'

habilidades_entrada = "sotware pmi project management"

# Inicializar el vectorizador TF-IDF
tfidf = TfidfVectorizer(stop_words='english')

# Concatenar las habilidades en una sola cadena
df['Skills_text'] = df['Skills'].apply(lambda x: ' '.join(x))

# Agregar las habilidades de entrada a una lista y luego al DataFrame
nuevo_empleo = pd.DataFrame({'Titulo': ['Habilidades de Entrada'], 'Skills_text': [habilidades_entrada]})
df_combined = pd.concat([df, nuevo_empleo], ignore_index=True)

# Aplicar TF-IDF al texto de habilidades
tfidf_matrix = tfidf.fit_transform(df_combined['Skills_text'])

# Calcular similitud del coseno entre las habilidades de entrada y los empleos
cosine_sim = linear_kernel(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

# Obtener los índices de los empleos más similares
related_jobs_indices = cosine_sim.argsort()[:-10:-1]  

# Imprimir los títulos de los empleos recomendados con la distancia de similitud
print("Empleos recomendados y su distancia de similitud:")
for i, index in enumerate(related_jobs_indices, start=1):
    title = df.iloc[index]['Titulo']
    link = df.iloc[index]['Link']
    skills = df.iloc[index]['Skills']
    
    # Calcular similitud de coseno entre habilidades de entrada y habilidades recomendadas
    tfidf_matrix_entrada = tfidf.transform([habilidades_entrada])
    tfidf_matrix_recomendada = tfidf.transform([df.iloc[index]['Skills']])
    similarity_score = linear_kernel(tfidf_matrix_entrada, tfidf_matrix_recomendada)[0][0]
    
    print(f"{i}. Titulo: {title}")
    print(f"   Link: {link}")
    print(f"   Skills: {skills}")
    print(f"   Distancia de similitud: {similarity_score}")
    print("\n")


# In[67]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

# Supongamos que tienes un DataFrame 'df' con columnas 'Titulo' y 'Skills'
# También se supone que ya tienes definida 'habilidades_entrada'

habilidades_entrada = "sotware pmi project management"

# Inicializar el vectorizador TF-IDF
tfidf = TfidfVectorizer(stop_words='english')

# Concatenar las habilidades en una sola cadena
df['Skills_text'] = df['Skills'].apply(lambda x: ' '.join(x))

# Agregar las habilidades de entrada a un nuevo DataFrame
nuevo_empleo = pd.DataFrame({'Titulo': ['Habilidades de Entrada'], 'Skills_text': [habilidades_entrada]})
df_combined = pd.concat([df, nuevo_empleo], ignore_index=True)

# Aplicar TF-IDF al texto de habilidades
tfidf_matrix = tfidf.fit_transform(df_combined['Skills_text'])

# Calcular similitud del coseno entre las habilidades de entrada y los empleos
cosine_sim = linear_kernel(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

# Obtener los índices de los empleos con similitud mayor a 0.5
threshold = 0.5
related_jobs_indices = [i for i, score in enumerate(cosine_sim) if score > threshold]

# Imprimir los títulos de los empleos recomendados con la distancia de similitud
print("Empleos recomendados con similitud mayor a 0.5:")
for index in related_jobs_indices:
    title = df.iloc[index]['Titulo']
    link = df.iloc[index]['Link']
    skills = df.iloc[index]['Skills']
    
    # Calcular similitud de coseno entre habilidades de entrada y habilidades recomendadas
    tfidf_matrix_entrada = tfidf.transform([habilidades_entrada])
    tfidf_matrix_recomendada = tfidf.transform([df.iloc[index]['Skills']])
    similarity_score = linear_kernel(tfidf_matrix_entrada, tfidf_matrix_recomendada)[0][0]
    
    print(f"Titulo: {title}")
    print(f"Link: {link}")
    print(f"Skills: {skills}")
    print(f"Distancia de similitud: {similarity_score}")
    print("\n")


# In[43]:


df['Skills'].iloc[1]


# In[3]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd


# Unir todas las habilidades en una sola cadena
text = ' '.join(df['Skills'])

# Generar la nube de palabras
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Mostrar la nube de palabras
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[ ]:





# In[59]:





# In[ ]:




