import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Cargar y procesar los datos
url = 'https://github.com/CabreraJDev/despliege_modelo/blob/main/datos_power_bi.csv'
df_completo = pd.read_csv('datos_power_bi.csv')

X = df_completo['comentarios_procesado']
y = df_completo['encoding']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

# Vectorizar las palabras
vectorizer = TfidfVectorizer(max_features=2000)
X_train_tfidf = vectorizer.fit_transform(X_train)

# Entrenar el modelo
model = LogisticRegression(C=10, penalty='l2', solver='lbfgs')
model.fit(X_train_tfidf, y_train)

# Función para predecir la clasificación de un nuevo comentario
def predecir_clasificacion(comentario):
    # Vectorizar el comentario
    comentario_tfidf = vectorizer.transform([comentario])
    
    # Predecir la clase
    prediccion = model.predict(comentario_tfidf)
    
    if prediccion == 1:
        return 'Neutral'
    if prediccion == 0:
        return 'Negativo'
    if prediccion == 2:
        return 'Positivo'
    

# Interfaz de Streamlit
st.title("Clasificación de Comentarios")
st.write("El modelo utilizando es la Regresión Logística.")

comentario_nuevo = st.text_input("Introduce un comentario:")
if st.button("Predecir clasificación"):
    clasificacion = predecir_clasificacion(comentario_nuevo)
    
    st.write(f"La clasificación del comentario es: {clasificacion}")
