#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import pandas as pd
archivo_csv = "C://Users//ASUS//Documents//SOFIA//IA//DATA//Linkedin_TI_2k_SV.csv"

df = pd.read_csv('Linkedin_TI12k.csv')


# In[3]:


df.head()


# In[7]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

# Suponiendo que 'Titulo' es el nombre de la columna en tu DataFrame 'df'
# Reemplaza 'df' con el nombre real de tu DataFrame y 'Titulo' con el nombre real de tu columna
text = ' '.join(df['Titulo'].dropna().astype(str).tolist())

# Configuración de la WordCloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Mostrar la nube de palabras
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[8]:


import pandas as pd

# Suponiendo que 'Sectores' es el nombre de la columna en tu DataFrame 'df'
# Reemplaza 'df' con el nombre real de tu DataFrame y 'Sectores' con el nombre real de tu columna
df = df[df['Sectores'] != 'Not specified']


# In[11]:


import matplotlib.pyplot as plt
import pandas as pd

# Suponiendo que 'Sectores' es el nombre de la columna en tu DataFrame 'df'
# Reemplaza 'df' con el nombre real de tu DataFrame y 'Sectores' con el nombre real de tu columna
sectores_count = df['Sectores'].value_counts()

# Configuración del gráfico de barras
plt.figure(figsize=(10, 6))
sectores_count.plot(kind='bar', color='skyblue')
plt.title('Cantidad de datos por sector')
plt.xlabel('Sector')
plt.ylabel('Cantidad de datos')
plt.xticks(rotation=45)  # Rotar etiquetas del eje x para mejor legibilidad si es necesario
plt.grid(axis='y')  # Agregar líneas de la cuadrícula en el eje y
plt.tight_layout()

# Mostrar el gráfico
plt.show()


# In[13]:


import matplotlib.pyplot as plt
import pandas as pd

# Suponiendo que 'Sectores' es el nombre de la columna en tu DataFrame 'df'
# Reemplaza 'df' con el nombre real de tu DataFrame y 'Sectores' con el nombre real de tu columna
sectores_count = df['Sectores'].value_counts()

# Seleccionar los 10 primeros sectores con más datos
top_10_sectores = sectores_count.head(10)

# Configuración del gráfico de pastel para los 10 primeros sectores
plt.figure(figsize=(8, 8))
plt.pie(top_10_sectores, labels=top_10_sectores.index, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('Top 10 Sectores con más datos')

# Mostrar el gráfico
plt.show()


# In[7]:


print('Dataframe shape is {}, containing a total of {} jobs'.format(df.shape,df.shape[0]))


# In[8]:


df_teams = df.groupby(by=['Sectores']).count().reset_index()
df_teams.sort_values(by='Titulo', ascending=False, inplace=True)


# In[31]:


import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.barplot(data=df_teams, x='Titulo', y='Sectores')
blue_l = "#ADD8E6"
#Highlight
for bar in ax.patches:
    if bar.get_y() < 0:
        bar.set_color(blue_l)  

#Labels
plt.xlabel('Count', fontsize=16)
plt.ylabel('Teams', fontsize=16)
plt.title('Amount of Jobs per Team', fontsize=20, pad=20);


# In[9]:


print('Out of a total of {} jobs, {:.2f}% ({}) are in not specified'.format(df['Sectores'].value_counts().sum(),((df['Sectores'].value_counts()[0]/df['Sectores'].value_counts().sum()) * 100),df['Sectores'].value_counts()[0]))


# In[33]:


###Vamos a saber cuantas descripciones vacias tenemos
# Suponiendo que tu DataFrame se llama df y la columna es 'Cuerpo'
count_description_available = df[df['Cuerpo'].str.contains('description available', case=False, na=False)].shape[0]

print(f"La cantidad de filas con 'description available' en la columna 'Cuerpo' es: {count_description_available}")


# In[15]:


print(12438-6390)


# #### UTILIZAREMOS PNL PARA INFERIR DESCRIPCIONES VACIAS

# ### Los modelos de lenguaje generativo son modelos capaces de generar texto continuo que se asemeje al lenguaje humano, usaremos RNNs, LSTM o GRU.
# 
# #### Aqui usaremos un modelo de lenguaje generativo basico utilizando tensorflow y keras para una arquitectura LSTM

# In[36]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Supongamos que tienes un conjunto de datos en una lista llamada 'text_data'
# Aquí 'text_data' contiene textos concatenados, no columnas separadas para título y descripción
# Puedes adaptar esto según la estructura de tus datos

# Preprocesamiento del texto
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_data)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in text_data:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Creación de features (X) y etiquetas (y) para entrenamiento
X, y = input_sequences[:,:-1],input_sequences[:,-1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Definición del modelo LSTM
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(LSTM(150, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

# Compilación del modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenamiento del modelo
model.fit(X, y, epochs=100, verbose=1)


# In[37]:


get_ipython().system('pip install openai')


# In[38]:


import openai

# Configurar la API key de OpenAI
api_key = 'tu_api_key'  # Reemplazar con tu propia API key
openai.api_key = api_key

# Obtener descripciones para las filas que contienen 'not available'
for idx in indices_not_available:
    description = X[idx]['Cuerpo']
    
    # Generar descripción utilizando GPT-3
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Complete the description: '{description}' ",
        max_tokens=50  # Ajustar según la longitud deseada
    )
    
    generated_description = response.choices[0].text.strip()
    print(f"Fila {idx}: {generated_description}")


# In[42]:


get_ipython().system('pip install transformers')


# In[49]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense
import numpy as np

# Suponiendo que 'df' es tu DataFrame con la columna 'Cuerpo'

# Obtener textos existentes para entrenar el modelo
texts = df['Cuerpo'].astype(str).tolist()

# Tokenizar y convertir a secuencias
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Preparar datos para el modelo LSTM
sequences = np.array(sequences)
X, y = sequences[:, :-1], sequences[:, -1]
vocab_size = len(tokenizer.word_index) + 1

# Definir modelo LSTM
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=X.shape[1]))
model.add(LSTM(100))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=50, batch_size=128)

# Generar textos similares
def generate_similar_text(seed_text, next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=X.shape[1], padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Texto de semilla para generar
seed_text = "texto de semilla"

# Generar texto similar
generated_text = generate_similar_text(seed_text, 50)
print(generated_text)


# In[47]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[26]:


###Vamos a saber cuantas descripciones vacias tenemos
# Suponiendo que tu DataFrame se llama df y la columna es 'Cuerpo'
count_description_available = df[df['Cuerpo'].str.contains('description available', case=False, na=False)].shape[0]

print(f"La cantidad de filas con 'No description available' en la columna 'Cuerpo' es: {count_description_available}")


# ### PROCESAMIENTO DE DATOS

# In[10]:


import pandas as pd

# Verificar datos faltantes
datos_faltantes = df.isnull()

# Contar datos faltantes por columna
conteo_faltantes = datos_faltantes.sum()

# Mostrar el conteo de datos faltantes
print(conteo_faltantes)


# ### ANALISIS DE DATOS

# In[11]:


###Observando la cantidad de datos 
print(len(df))


# In[12]:


import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Concatenar todas las descripciones en un solo texto
text = ' '.join(df['Cuerpo'])

# Crear un objeto WordCloud con colores neutros
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    colormap='gray',  # Cambia el colormap al que prefieras (por ejemplo, 'gray' para colores grises)
    collocations=False  # Deshabilitar la detección de frases coloquiales
).generate(text)

# Mostrar la nube de palabras clave en una figura de matplotlib
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Nube de Palabras Clave en Descripciones de Trabajo')
plt.show()


# In[13]:


from wordcloud import WordCloud
import pandas as pd
from stop_words import get_stop_words
from nltk.tokenize import word_tokenize

stop_words = get_stop_words('es')  # Ajusta el código del idioma según tus necesidades

# Concatenar el texto de la columna "Cuerpo" en una sola cadena
text = ' '.join(df['Cuerpo'])

# Tokenizar el texto
words = word_tokenize(text)

# Filtrar palabras que no sean preposiciones
filtered_words = [word.lower() for word in words if word.lower() not in stop_words]

# Unir las palabras filtradas en una sola cadena
filtered_text = ' '.join(filtered_words)

# Crear la nube de palabras
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(filtered_text)

# Mostrar la nube de palabras (opcional)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[20]:


##Eliminando todas las filas que no tienen datos en la columna CUERPO
# Supongamos que tu DataFrame se llama df
import pandas as pd

# Supongamos que tu DataFrame se llama df
df = df[~df['Cuerpo'].str.contains('No description available', case=False, na=False)]

# Imprimir el DataFrame resultante
print(df)


# In[21]:


len(df)


# ### Observando cual es el país con más datos

# In[22]:


import pandas as pd


# Función para obtener el país después de las dos comas
def obtener_pais(localizacion):
    partes = localizacion.split(",")  # Dividir la cadena en partes separadas por comas
    if len(partes) > 2:
        return partes[-1].strip()  # Tomar la última parte y quitar espacios en blanco al inicio o final

# Aplicar la función para extraer el país
df['Pais'] = df['Localizacion'].apply(obtener_pais)

# Contar la cantidad de países únicos
cantidad_paises = df['Pais'].nunique()

print(f"La cantidad de países distintos en la columna 'Localizacion' es: {cantidad_paises}")


# In[23]:


import pandas as pd
import matplotlib.pyplot as plt

# Supongamos que ya has aplicado la función y creado la columna 'Pais' en tu DataFrame 'df'

# Count de la cantidad de países
conteo_paises = df['Pais'].value_counts()

# Ordenar de mayor a menor
conteo_paises = conteo_paises.sort_values(ascending=False)

# Gráfico de barras
plt.figure(figsize=(10, 6))  # Tamaño del gráfico
conteo_paises.plot(kind='bar')
plt.title('Cantidad de países en la columna "Localizacion"')
plt.xlabel('Países')
plt.ylabel('Cantidad')
plt.show()


# ### Preprocess text data

# In[24]:


df['Cuerpo'] = df['Cuerpo'].apply(lambda x: " ".join(x.lower()for x in x.split()))
## remove tabulation and punctuation
df['Cuerpo'] = df['Cuerpo'].str.replace('[^\w\s]',' ')
## digits
df['Cuerpo'] = df['Cuerpo'].str.replace('\d+', '')


# ### Stop Words

# In[25]:


import pandas as pd
import chardet

# Ruta al archivo de stopwords
ruta_stopwords = 'C://Users//ASUS//Documents//SOFIA//IA//DATA//Stopwords//stopwords.txt'

# Detectar la codificación del archivo
with open(ruta_stopwords, 'rb') as file:
    data = file.read()
    codificacion = chardet.detect(data)['encoding']

# Cargar el archivo de stopwords con la codificación detectada
with open(ruta_stopwords, 'r', encoding=codificacion) as file:
    stopwords_list = file.read().splitlines()

# Suponiendo que tienes un DataFrame 'df' con una columna 'Cuerpo'

# Función para eliminar stopwords
def eliminar_stopwords(texto):
    palabras = texto.split()
    palabras_sin_stopwords = [palabra for palabra in palabras if palabra.lower() not in stopwords_list]
    return ' '.join(palabras_sin_stopwords)

# Aplicar la eliminación de stopwords a la columna 'Cuerpo'
df['Cuerpo_sin_stopwords'] = df['Cuerpo'].apply(eliminar_stopwords)


# In[26]:


## jda stands for job description aggregated
jda = df.groupby(['Titulo']).sum().reset_index()
print("Aggregated job descriptions: \n")
print(jda)


# In[27]:


jobs_list = jda.Titulo.unique().tolist()
for job in jobs_list:

    # Start with one review:
    text = jda[jda.Titulo == job].iloc[0].Cuerpo_sin_stopwords
    # Create and generate a word cloud image:
    wordcloud = WordCloud().generate(text)
    print("\n***",job,"***\n")
    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


# In[28]:


import pandas as pd
import string
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('spanish'))  # Lista de palabras vacías en español

# Función para eliminar palabras vacías
def remove_stopwords(text):
    words = text.split()  # Separa el texto en palabras
    clean_words = [word for word in words if word.lower() not in stop_words]  # Elimina palabras vacías
    return ' '.join(clean_words)  # Une las palabras limpias de nuevo en un texto

# Aplica la función a la columna 'Cuerpo'
df['Cuerpo_sin_stopwords'] = df['Cuerpo_sin_stopwords'].apply(remove_stopwords)


# ### MODELAMIENTO
# ### Convertir el texto en caracteristicas

# In[29]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

vectorizer = TfidfVectorizer()
#Tokenize and build vocabulary
X = vectorizer.fit_transform(df.Cuerpo_sin_stopwords)
y = df.Titulo

# split data into 80% training and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=109) 
print("train data shape: ",X_train.shape)
print("test data shape: ",X_test.shape)

# Fit model
clf = MultinomialNB()
clf.fit(X_train, y_train)
## Predict
y_predicted = clf.predict(X_test)


# In[30]:


y_train.hist()
y_test.hist()


# In[31]:


df.head()


# ### Entrenamiento del modelo, probando con SVM, Naive Bayes, Random Forest o Redes neuronales

# ### STOP WORDS

# In[32]:


import pandas as pd
import chardet

# Ruta al archivo de stopwords
ruta_stopwords = 'C://Users//ASUS//Documents//SOFIA//IA//DATA//Stopwords//stopwords.txt'

# Detectar la codificación del archivo
with open(ruta_stopwords, 'rb') as file:
    data = file.read()
    codificacion = chardet.detect(data)['encoding']

# Cargar el archivo de stopwords con la codificación detectada
with open(ruta_stopwords, 'r', encoding=codificacion) as file:
    stopwords_list = file.read().splitlines()

# Suponiendo que tienes un DataFrame 'df' con una columna 'Cuerpo'

# Función para eliminar stopwords
def eliminar_stopwords(texto):
    palabras = texto.split()
    palabras_sin_stopwords = [palabra for palabra in palabras if palabra.lower() not in stopwords_list]
    return ' '.join(palabras_sin_stopwords)

# Aplicar la eliminación de stopwords a la columna 'Cuerpo'
df['Cuerpo_sin_stopwords'] = df['Cuerpo'].apply(eliminar_stopwords)


# In[33]:


print(df['Cuerpo_sin_stopwords'])


# In[34]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Convertir la columna 'Cuerpo_sin_stopwords' a una lista de texto
texto = ' '.join(df['Cuerpo_sin_stopwords'].dropna().tolist())

# Crear un objeto WordCloud con las configuraciones deseadas
wordcloud = WordCloud(width=800, height=400, max_words=200, background_color='white').generate(texto)

# Mostrar la nube de palabras usando matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Nube de palabras - Columna Cuerpo_sin_stopwords')
plt.show()


# In[35]:


print(df["Cuerpo"])


# In[36]:


import pandas as pd
from gensim import corpora, models
import gensim
from nltk.corpus import stopwords
import string

# Definir la función de preprocesamiento
def preprocess_text(words):
    stop_words = set(stopwords.words('spanish'))
    punctuation = string.punctuation
    words = [word.lower() for word in words if word not in stop_words and word not in punctuation]
    return ' '.join(words)  # Convertir la lista de palabras a una cadena de texto


# Aplicar la función de preprocesamiento al DataFrame
df['Processed_Cuerpo'] = df['Cuerpo'].apply(preprocess_text)

# Crear un diccionario y un corpus
dictionary = corpora.Dictionary(df['Processed_Cuerpo'].apply(str.split))  # Se convierten las cadenas a listas de palabras
corpus = [dictionary.doc2bow(text.split()) for text in df['Processed_Cuerpo']]

# Aplicar el modelo LDA
num_topics = 5
lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary)

# Asignar cada texto a un tópico
df['Topico'] = df['Processed_Cuerpo'].apply(lambda x: lda_model[dictionary.doc2bow(x.split())][0][0])

# Imprimir los textos por tópico
for topic in range(num_topics):
    topic_texts = df[df['Topico'] == topic]['Cuerpo']
    print(f'Tópico {topic}:\n')
    for text in topic_texts:
        print(text)
    print('\n')


# #### ALGORITMO TF-IDF (Term Frequency-Inverse Document Frequency)

# In[37]:


from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


# Asegurémonos de que los datos en 'Cuerpo' sean cadenas (strings)
df['Cuerpo_sin_stopwords'] = df['Cuerpo_sin_stopwords'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)

# Crear una instancia de TfidfVectorizer
tfidf = TfidfVectorizer()

# Calcular TF-IDF sobre todas las descripciones de trabajo
tfidf_matrix = tfidf.fit_transform(df['Cuerpo_sin_stopwords'])

# Obtener los nombres de las características (palabras)
feature_names = tfidf.get_feature_names_out()

# Calcular los puntajes TF-IDF para todo el conjunto de descripciones
tfidf_scores = tfidf_matrix.toarray()

# Crear una columna con las palabras clave para cada fila y agregarlas como una nueva columna 'PalabrasClave'
keywords_per_row = []
for row in tfidf_scores:
    keywords = " ".join([feature_names[ind] for ind in row.argsort()[-5:][::-1]])  # Obtener las 5 palabras clave con los puntajes más altos
    keywords_per_row.append(keywords)

# Agregar las palabras clave al DataFrame como una nueva columna 'PalabrasClave'
df['PalabrasClave'] = keywords_per_row

# Imprimir las primeras filas del DataFrame con la nueva columna 'PalabrasClave'
print(df.head())


# In[38]:


df.head()


# In[40]:


# Imprimir las palabras clave de la primera fila
palabras_clave_primera_fila = df.loc[1, 'PalabrasClave']
print(palabras_clave_primera_fila)


# In[41]:


import spacy
import pandas as pd
# Cargar el modelo pre-entrenado en español de spaCy
nlp = spacy.load("es_core_news_sm")  # Puedes elegir un modelo más grande si es necesario


# Suponiendo que tienes un DataFrame df con una columna 'Cuerpo' que contiene las descripciones de trabajo

# Función para analizar y encontrar entidades por categorías
def analizar_categorias(texto):
    doc = nlp(texto)
    categorias = {ent.label_: [ent.text] for ent in doc.ents}
    return categorias

# Aplicar el análisis por categorías a la columna 'Cuerpo'
df['Categorias'] = df['Cuerpo_sin_stopwords'].apply(analizar_categorias)

# Mostrar el resultado para la primera fila
print(df['Categorias'].iloc[0])


# In[42]:


df.head()


# In[43]:


df.info()


# In[44]:


from sklearn.feature_extraction.text import TfidfVectorizer

tdif = TfidfVectorizer()

df['Cuerpo_sin_stopwords'] = df['Cuerpo_sin_stopwords'].fillna('')

tdif_matrix = tdif.fit_transform(df["Cuerpo_sin_stopwords"])

tdif_matrix.shape

from sklearn.metrics.pairwise import sigmoid_kernel

cosine_sim = sigmoid_kernel(tdif_matrix, tdif_matrix)
indices = pd.Series(df.index, index=df['Titulo']).drop_duplicates()


# In[45]:


def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda X: X[1], reverse=True)
    sim_scores = sim_scores[1:16]
    tech_indices = [i[0] for i in sim_scores]
    return df["Titulo"].iloc[tech_indices]


# In[46]:


new1 = df[['Titulo','Cuerpo']]


# In[63]:


df.tail()


# In[65]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Suponiendo que tienes un DataFrame 'data' con una columna 'Titulo'

# Crear un objeto TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Titulo'])

# Entrenar el modelo de agrupamiento K-Means
k = 5  # Número de clusters deseado
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)

# Asignar los clusters a los datos
df['Categoria_kmeans'] = kmeans.labels_

# Mostrar los títulos agrupados por categorías generadas automáticamente
for i in range(k):
    print(f"Categoría {i}:")
    print(df[df['Categoria_kmeans'] == i]['Titulo'])
    print("\n")


# In[66]:


df.head()


# ### Modelando la DATA

# In[67]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

vectorizer = TfidfVectorizer()
#Tokenize and build vocabulary
X = vectorizer.fit_transform(df.Cuerpo_sin_stopwords)
y = df.Categoria_kmeans

# split data into 80% training and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=109) 
print("train data shape: ",X_train.shape)
print("test data shape: ",X_test.shape)

# Fit model
clf = MultinomialNB()
clf.fit(X_train, y_train)
## Predict
y_predicted = clf.predict(X_test)


# In[68]:


y_train.hist()
y_test.hist()


# In[69]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
#evaluate the predictions
print("Accuracy score is: ",accuracy_score(y_test, y_predicted))
#print("Classes: (to help read Confusion Matrix)\n", clf.classes_)
print("Confusion Matrix: ")

print(confusion_matrix(y_test, y_predicted))
print("Classification Report: ")
print(classification_report(y_test, y_predicted))


# In[57]:


df.head()


# In[61]:


print(clf.feature_log_prob_)
print(clf.feature_log_prob_.shape)


# In[73]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

vectorizer = TfidfVectorizer()
# Tokenizar y construir el vocabulario
X = vectorizer.fit_transform(df.Cuerpo_sin_stopwords)
y = df.Titulo

# dividir los datos en 80% entrenamiento y 20% prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=109) 
print("Forma de los datos de entrenamiento: ", X_train.shape)
print("Forma de los datos de prueba: ", X_test.shape)

# Ajustar el modelo
clf = MultinomialNB()
clf.fit(X_train, y_train)

## Predecir
y_predicted = clf.predict(X_test)



# In[81]:


get_ipython().system('pip install textblob')
get_ipython().system('python -m textblob.download_corpora')


# In[82]:


import numpy as np
from textblob import TextBlob
technical_skills = ['python', 'c','r', 'c++','java','hadoop','scala','flask','pandas','spark','scikit-learn',
                    'numpy','php','sql','mysql','css','mongdb','nltk','fastai' , 'keras', 'pytorch','tensorflow',
                    'linux','Ruby','JavaScript','django','react','reactjs','ai','ui','tableau']

# Supongamos que tienes un TfidfVectorizer llamado 'vectorizer' que ya fue ajustado a tus datos
feature_array = vectorizer.get_feature_names_out()

# Número total de características en el modelo
features_numbers = len(feature_array)

# Número máximo de características ordenadas
n_max = int(features_numbers * 0.1)

# Inicializar el dataframe de salida
output = pd.DataFrame()

for i in range(0, len(clf.classes_)):
    print("\n****" ,clf.classes_[i],"****\n")
    class_prob_indices_sorted = clf.feature_log_prob_[i, :].argsort()[::-1]
    raw_skills = np.take(feature_array, class_prob_indices_sorted[:n_max])
    print("Lista de habilidades no procesadas:")
    print(raw_skills)
    
    # Extraer habilidades técnicas
    top_technical_skills = list(set(technical_skills).intersection(raw_skills))[:6]

    # Extraer adjetivos
    txt = " ".join(raw_skills)
    blob = TextBlob(txt)
    top_adjectives = [w for (w, pos) in TextBlob(txt).pos_tags if pos.startswith("JJ")][:6]
    
    output = output.append({'Titulo': clf.classes_[i],
                            'technical_skills': top_technical_skills,
                            'soft_skills': top_adjectives},
                           ignore_index=True)


# In[107]:


import numpy as np
from textblob import TextBlob
import pandas as pd

technical_skills = ['python', 'c','r', 'c++','java','hadoop','scala','flask','pandas','spark','scikit-learn',
                    'numpy','php','sql','mysql','css','mongdb','nltk','fastai' , 'keras', 'pytorch','tensorflow',
                    'linux','Ruby','JavaScript','django','react','reactjs','ai','ui','tableau']

# Supongamos que tienes un TfidfVectorizer llamado 'vectorizer' que ya fue ajustado a tus datos
feature_array = vectorizer.get_feature_names_out()

# Número total de características en el modelo
features_numbers = len(feature_array)

# Número máximo de características ordenadas
n_max = int(features_numbers * 0.1)

# Inicializar la lista de salida
output_data = []

for i in range(0, len(clf.classes_)):
    print("\n****" ,clf.classes_[i],"****\n")
    class_prob_indices_sorted = clf.feature_log_prob_[i, :].argsort()[::-1]
    raw_skills = np.take(feature_array, class_prob_indices_sorted[:n_max])
    print("Lista de habilidades no procesadas:")
    print(raw_skills)
    
    # Extraer habilidades técnicas
    top_technical_skills = list(set(technical_skills).intersection(raw_skills))[:6]

    # Extraer adjetivos
    txt = " ".join(raw_skills)
    blob = TextBlob(txt)
    top_adjectives = [w for (w, pos) in TextBlob(txt).pos_tags if pos.startswith("JJ")][:6]
    
    output_data.append({'Categoria_kmeans': clf.classes_[i],
                        'technical_skills': top_technical_skills,
                        'soft_skills': top_adjectives})
    print(output['technical_skills'])
# Crear el DataFrame a partir de la lista de diccionarios
#output = pd.DataFrame(output_data)

df['skills_data'] = output['technical_skills']
df['soft_skills'] = output['soft_skills']


# In[118]:


import pandas as pd

data = []

for i in range(0, len(clf.classes_)):
    print("\n****", clf.classes_[i], "****\n")
    class_prob_indices_sorted = clf.feature_log_prob_[i, :].argsort()[::-1]
    raw_skills = np.take(feature_array, class_prob_indices_sorted[:n_max])
    #print("Lista de habilidades no procesadas:")
    #print(raw_skills)
    
    # Asegurémonos de que la lista de habilidades técnicas se ajuste a cada clase
    top_technical_skills = list(set(technical_skills).intersection(raw_skills))[:6]

    # Extraer adjetivos
    txt = " ".join(raw_skills)
    blob = TextBlob(txt)
    top_adjectives = [w for (w, pos) in TextBlob(txt).pos_tags if pos.startswith("JJ")][:6]
    
    data.append({'Titulo': clf.classes_[i],
                 'technical_skills': top_technical_skills,
                 'soft_skills': top_adjectives})

output = pd.DataFrame(data)

print(output['technical_skills'])
df['skills_data'] = output['technical_skills']


# In[119]:


df.head(50)


# In[ ]:




