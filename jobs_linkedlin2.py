#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import pandas as pd
archivo_csv = "SOFIA//IA//DATA//Linkedin_TI_2k_SV.csv"

df = pd.read_csv('Linkedin_TI12k.csv')


# In[3]:

df.head()

# In[7]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

text = ' '.join(df['Titulo'].dropna().astype(str).tolist())


wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Mostrar la nube de palabras
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[8]:


import pandas as pd

df = df[df['Sectores'] != 'Not specified']


# In[11]:


import matplotlib.pyplot as plt
import pandas as pd


sectores_count = df['Sectores'].value_counts()

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

api_key = 'tu_api_key'  # Reemplazar con tu propia API key
openai.api_key = api_key

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


texts = df['Cuerpo'].astype(str).tolist()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

sequences = np.array(sequences)
X, y = sequences[:, :-1], sequences[:, -1]
vocab_size = len(tokenizer.word_index) + 1

model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=X.shape[1]))
model.add(LSTM(100))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=50, batch_size=128)

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



# In[26]:

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
    colormap='gray', 
    collocations=False 
).generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Nube de Palabras Clave en Descripciones de Trabajo')
plt.show()



# ### Stop Words

# In[25]:


import pandas as pd
import chardet

# Ruta al archivo de stopwords
ruta_stopwords = 'SOFIA//IA//DATA//Stopwords//stopwords.txt'

with open(ruta_stopwords, 'rb') as file:
    data = file.read()
    codificacion = chardet.detect(data)['encoding']

with open(ruta_stopwords, 'r', encoding=codificacion) as file:
    stopwords_list = file.read().splitlines()


def eliminar_stopwords(texto):
    palabras = texto.split()
    palabras_sin_stopwords = [palabra for palabra in palabras if palabra.lower() not in stopwords_list]
    return ' '.join(palabras_sin_stopwords)

df['Cuerpo_sin_stopwords'] = df['Cuerpo'].apply(eliminar_stopwords)


# In[26]:


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

# In[107]:


import numpy as np
from textblob import TextBlob
import pandas as pd

technical_skills = ['python', 'c','r', 'c++','java','hadoop','scala','flask','pandas','spark','scikit-learn',
                    'numpy','php','sql','mysql','css','mongdb','nltk','fastai' , 'keras', 'pytorch','tensorflow',
                    'linux','Ruby','JavaScript','django','react','reactjs','ai','ui','tableau']

feature_array = vectorizer.get_feature_names_out()

features_numbers = len(feature_array)

n_max = int(features_numbers * 0.1)

output_data = []

for i in range(0, len(clf.classes_)):
    print("\n****" ,clf.classes_[i],"****\n")
    class_prob_indices_sorted = clf.feature_log_prob_[i, :].argsort()[::-1]
    raw_skills = np.take(feature_array, class_prob_indices_sorted[:n_max])
    print("Lista de habilidades no procesadas:")
    print(raw_skills)
    

    top_technical_skills = list(set(technical_skills).intersection(raw_skills))[:6]

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


# In[119]:

df.head(50)




