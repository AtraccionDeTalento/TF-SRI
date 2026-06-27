"""
Sistema de Recuperación de Información (SRI) — Versión auto-contenida
Todo el código de los módulos está inlinado aquí para que Streamlit Cloud
pueda ejecutarlo sin necesitar la carpeta modulos/ en el repositorio.
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import re
import os
import time
import unicodedata
import requests
from pathlib import Path
from collections import defaultdict

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO: PREPROCESAMIENTO NLP
# ══════════════════════════════════════════════════════════════════════════════

STOPWORDS_ES = {
    'a', 'al', 'algo', 'algunas', 'algunos', 'ante', 'antes', 'como',
    'con', 'contra', 'cual', 'cuando', 'de', 'del', 'desde', 'donde',
    'durante', 'e', 'el', 'ella', 'ellas', 'ellos', 'en', 'entre',
    'era', 'esa', 'esas', 'ese', 'eso', 'esos', 'esta', 'estaba',
    'estado', 'estar', 'estas', 'este', 'esto', 'estos', 'fue',
    'fueron', 'ha', 'hacia', 'hasta', 'hay', 'la', 'las', 'le',
    'les', 'lo', 'los', 'mas', 'me', 'mi', 'muy', 'na', 'ni', 'no',
    'nos', 'nosotros', 'o', 'otra', 'otras', 'otro', 'otros', 'para',
    'pero', 'por', 'que', 'quien', 'se', 'ser', 'si', 'sin', 'sino',
    'sobre', 'somos', 'son', 'su', 'sus', 'también', 'te', 'ti',
    'tiene', 'tienen', 'toda', 'todas', 'todo', 'todos', 'tu', 'tus',
    'un', 'una', 'unas', 'uno', 'unos', 'usted', 'ustedes', 'y', 'ya',
    'yo', 'es', 'fue', 'ser', 'está', 'más', 'sí', 'así', 'aquí',
    'ahí', 'aún', 'hoy', 'bien', 'puede', 'cada', 'hace', 'han',
    'hay', 'he', 'ir', 'va', 'vez', 'dos', 'tres', 'ser'
}


def _stemmer_es(palabra):
    """Stemming básico para español usando reglas de sufijos."""
    if len(palabra) <= 4:
        return palabra
    sufijos = [
        'amientos', 'imientos', 'amiento', 'imiento',
        'aciones', 'uciones', 'idades',
        'mente', 'acion', 'ucion', 'idad', 'ivos',
        'ivas', 'ista', 'ismo', 'able', 'ible',
        'ando', 'endo', 'aron', 'ieron', 'aban',
        'rian', 'ados', 'idos', 'ante', 'ores',
        'cion', 'dad', 'ivo', 'iva', 'oso', 'osa',
        'ado', 'ido', 'nte', 'les', 'ces', 'or',
        'ar', 'er', 'ir', 'as', 'es', 'os'
    ]
    for sufijo in sufijos:
        if palabra.endswith(sufijo) and len(palabra) - len(sufijo) >= 3:
            return palabra[:-len(sufijo)]
    return palabra


def tokenizar(texto):
    """Tokeniza el texto: lowercase + split alfanumérico."""
    texto = texto.lower()
    texto = re.sub(r"[^a-záéíóúñü0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto.split()


def eliminar_stopwords(tokens):
    """Elimina stopwords del español."""
    return [t for t in tokens if t not in STOPWORDS_ES and len(t) > 1]


def aplicar_stemming(tokens):
    """Aplica stemming a cada token."""
    return [_stemmer_es(t) for t in tokens]


def preprocesar_simple(texto):
    """Pipeline simple sin stopwords ni stemming (para BM25 base)."""
    texto = texto.lower()
    texto = re.sub(r"[^a-záéíóúñü0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto.split()


def preprocesar_completo(texto, usar_stemming=True):
    """Pipeline completo: normalizar -> tokenizar -> stopwords -> stem."""
    texto_norm = unicodedata.normalize("NFKD", str(texto))
    texto_norm = texto_norm.encode("ascii", "ignore").decode("ascii")
    tokens = tokenizar(texto_norm)
    tokens = eliminar_stopwords(tokens)
    if usar_stemming:
        tokens = aplicar_stemming(tokens)
    return tokens


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO: MODELOS DE RECUPERACIÓN DE INFORMACIÓN
# ══════════════════════════════════════════════════════════════════════════════

class MotorBM25:
    """Motor de búsqueda basado en BM25 Okapi."""

    def __init__(self, corpus_textos, archivos):
        self.archivos = archivos
        self.corpus_tokenizado = [preprocesar_simple(t) for t in corpus_textos]
        self.modelo = BM25Okapi(self.corpus_tokenizado)
        self.num_docs = len(corpus_textos)

    def buscar(self, consulta, top_k=10):
        inicio = time.time()
        tokens_consulta = preprocesar_simple(consulta)
        scores = self.modelo.get_scores(tokens_consulta)
        indices = np.argsort(scores)[::-1][:top_k]
        tiempo = time.time() - inicio

        resultados = []
        for rank, idx in enumerate(indices, start=1):
            resultados.append({
                "rank": rank,
                "id_documento": int(idx),
                "archivo": self.archivos[idx],
                "score": float(scores[idx]),
            })
        return {
            "metodo": "BM25",
            "tiempo_ms": round(tiempo * 1000, 4),
            "total_docs": self.num_docs,
            "resultados": resultados,
            "scores_completos": scores,
        }


class MotorTFIDF:
    """Motor de búsqueda basado en TF-IDF + Similitud Coseno."""

    def __init__(self, corpus_textos, archivos):
        self.archivos = archivos
        self.num_docs = len(corpus_textos)
        self.vectorizador = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"\b[a-záéíóúñü0-9]+\b",
            max_features=5000,
            ngram_range=(1, 2),
        )
        self.matriz = self.vectorizador.fit_transform(corpus_textos)
        # Atributos expuestos para la visualización del índice invertido
        self.vocabulario = self.vectorizador.vocabulary_
        self.vocabulario_size = len(self.vocabulario)
        self.matriz_tfidf = self.matriz.toarray()

    def buscar(self, consulta, top_k=10):
        inicio = time.time()
        vector_consulta = self.vectorizador.transform([consulta])
        scores = cosine_similarity(vector_consulta, self.matriz)[0]
        indices = np.argsort(scores)[::-1][:top_k]
        tiempo = time.time() - inicio

        resultados = []
        for rank, idx in enumerate(indices, start=1):
            resultados.append({
                "rank": rank,
                "id_documento": int(idx),
                "archivo": self.archivos[idx],
                "score": float(scores[idx]),
            })
        return {
            "metodo": "TF-IDF",
            "tiempo_ms": round(tiempo * 1000, 4),
            "total_docs": self.num_docs,
            "resultados": resultados,
            "scores_completos": scores,
        }


class SistemaRI:
    """Sistema de Recuperación de Información que integra BM25 y TF-IDF."""

    def __init__(self, textos, archivos, palabras_por_doc):
        self.textos = textos
        self.archivos = archivos
        self.palabras_por_doc = palabras_por_doc
        self.motor_bm25 = MotorBM25(textos, archivos)
        self.motor_tfidf = MotorTFIDF(textos, archivos)

    def buscar_comparativo(self, consulta, top_k=10):
        return {
            "consulta": consulta,
            "bm25": self.motor_bm25.buscar(consulta, top_k),
            "tfidf": self.motor_tfidf.buscar(consulta, top_k),
        }


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO: SCRAPER BUMERAN / MOCK WEB
# ══════════════════════════════════════════════════════════════════════════════

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "es-PE,es;q=0.9",
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
}


def _get(url, timeout=15):
    """HTTP GET con manejo de errores."""
    try:
        r = requests.get(url, headers=_HEADERS, timeout=timeout)
        r.raise_for_status()
        return r
    except Exception:
        return None


def _scholar_url(nombre, especialidad=""):
    """Genera URL de búsqueda en Semantic Scholar."""
    query = f"{nombre} {especialidad}".strip().replace(" ", "+")
    return f"https://www.semanticscholar.org/search?q={query}&sort=Relevance"


PERFILES_DEMO = [
    {
        "nombre": "Carlos Raymundo Ibañez",
        "titulo": "Docente Investigador — Ciencia de Datos e IA",
        "empresa": "UPC — Universidad Peruana de Ciencias Aplicadas",
        "area": "Educación Superior",
        "ubicacion": "Monterrico, Lima",
        "anios": 15,
        "especialidad": "Inteligencia Artificial, Machine Learning, IoT, Industria 4.0",
        "descripcion": (
            "Doctor en Ingeniería de Sistemas. Docente investigador con más de 15 años "
            "de experiencia en Machine Learning, Deep Learning e IoT aplicado a industria. "
            "Autor de más de 80 publicaciones en Scopus. Python, TensorFlow, MATLAB. "
            "Miembro del Grupo de Investigación en Inteligencia Artificial de la UPC."
        ),
        "url": _scholar_url("Carlos Raymundo Ibañez", "machine learning UPC"),
        "fuente": "Demo - Perfiles Docentes PE",
    },
    {
        "nombre": "Willy Ugarte Rojas",
        "titulo": "Docente Investigador — Machine Learning y Minería de Datos",
        "empresa": "UPC — Universidad Peruana de Ciencias Aplicadas",
        "area": "Educación Superior",
        "ubicacion": "Monterrico, Lima",
        "anios": 12,
        "especialidad": "Machine Learning, Data Mining, Python, Ciencia de Datos, NLP",
        "descripcion": (
            "Doctor en Computer Science. Investigador y docente en Machine Learning, "
            "Minería de Datos y NLP. Publicaciones en revistas Scopus Q1. "
            "Experiencia en proyectos de analítica para sector financiero peruano. "
            "Cursos: Aprendizaje Automático, Ciencia de Datos, Procesamiento de Texto."
        ),
        "url": _scholar_url("Willy Ugarte", "machine learning data mining"),
        "fuente": "Demo - Perfiles Docentes PE",
    },
    {
        "nombre": "Jose Luis Arteaga Huaraca",
        "titulo": "Investigador — Procesamiento de Lenguaje Natural",
        "empresa": "UTEC — Universidad de Ingeniería y Tecnología",
        "area": "Educación Superior",
        "ubicacion": "Barranco, Lima",
        "anios": 8,
        "especialidad": "NLP, Deep Learning, Transformers, BERT, Recuperación de Información",
        "descripcion": (
            "Investigador en NLP con enfoque en textos en español: análisis de sentimientos, "
            "sistemas de recuperación de información, modelos de lenguaje. "
            "Publicaciones en conferencias internacionales. PyTorch, HuggingFace. "
            "Docente de cursos de IA y procesamiento de lenguaje natural."
        ),
        "url": _scholar_url("Jose Arteaga", "NLP procesamiento lenguaje natural"),
        "fuente": "Demo - Perfiles Docentes PE",
    },
    {
        "nombre": "Edgar Sifuentes Cutipa",
        "titulo": "Docente Investigador — Big Data y Sistemas Distribuidos",
        "empresa": "UNMSM — Universidad Nacional Mayor de San Marcos",
        "area": "Educación Superior",
        "ubicacion": "Cercado de Lima",
        "anios": 14,
        "especialidad": "Big Data, Hadoop, Spark, Bases de Datos NoSQL, Cloud Computing",
        "descripcion": (
            "Doctor en Informática. Docente en Sistemas de Bases de Datos, "
            "Big Data y Computación en la Nube. Investigador CONCYTEC. "
            "Proyectos con sector público peruano. "
            "Publicaciones en procesamiento distribuido de datos."
        ),
        "url": _scholar_url("Edgar Sifuentes", "big data sistemas distribuidos"),
        "fuente": "Demo - Perfiles Docentes PE",
    },
    {
        "nombre": "Laberiano Andrade Arenas",
        "titulo": "Docente — Inteligencia Artificial y Visión por Computadora",
        "empresa": "UTP — Universidad Tecnológica del Perú",
        "area": "Educación Superior",
        "ubicacion": "Lima, Perú",
        "anios": 7,
        "especialidad": "Computer Vision, Deep Learning, OpenCV, Redes Neuronales, Python",
        "descripcion": (
            "Investigador en Inteligencia Artificial y Visión Computacional. "
            "Publicaciones en IEEE Access y Scopus sobre detección de objetos "
            "y clasificación de imágenes con CNN. Experiencia en Python, "
            "TensorFlow, Keras, OpenCV. Docente de IA en pregrado y posgrado."
        ),
        "url": _scholar_url("Laberiano Andrade", "computer vision deep learning"),
        "fuente": "Demo - Perfiles Docentes PE",
    },
    {
        "nombre": "Juan Carlos Gutierrez Caceres",
        "titulo": "Docente — Algoritmos y Programación",
        "empresa": "USIL — Universidad San Ignacio de Loyola",
        "area": "Educación Superior",
        "ubicacion": "La Molina, Lima",
        "anios": 6,
        "especialidad": "Algoritmos, Python, Java, Estructuras de Datos, Programación",
        "descripcion": (
            "Docente con experiencia en cursos de Programación, "
            "Algoritmos y Estructuras de Datos. Metodologías activas. "
            "Experiencia profesional en desarrollo de software. "
            "Python, Java, C++. Proyectos de software educativo."
        ),
        "url": _scholar_url("Juan Carlos Gutierrez", "algoritmos programacion"),
        "fuente": "Demo - Perfiles Docentes PE",
    },
    {
        "nombre": "Alex Pacheco Pumaleque",
        "titulo": "Docente — Business Intelligence y Analítica",
        "empresa": "UPN — Universidad Privada del Norte",
        "area": "Tecnología de la Información",
        "ubicacion": "Los Olivos, Lima",
        "anios": 9,
        "especialidad": "Business Intelligence, Power BI, SQL, Data Warehouse, ETL",
        "descripcion": (
            "Investigador en Business Intelligence y analítica de negocios. "
            "Docente de BI, Power BI, SQL Avanzado y Visualización de Datos. "
            "Publicaciones en gestión de datos empresariales. "
            "Certificación Microsoft Power BI Data Analyst."
        ),
        "url": _scholar_url("Alex Pacheco", "business intelligence analytics Peru"),
        "fuente": "Demo - Perfiles Docentes PE",
    },
    {
        "nombre": "Christian Vargas Cuentas",
        "titulo": "Investigador — Sistemas de Recuperación de Información",
        "empresa": "PUCP — Pontificia Universidad Católica del Perú",
        "area": "Educación Superior",
        "ubicacion": "San Miguel, Lima",
        "anios": 10,
        "especialidad": "Information Retrieval, BM25, TF-IDF, Indexación, Motores de Búsqueda",
        "descripcion": (
            "Investigador en recuperación de información y procesamiento de texto. "
            "Modelos BM25 y TF-IDF para búsqueda de documentos. "
            "Publicaciones en revistas indexadas sobre IR y NLP. "
            "Docente de posgrado en Recuperación de Información y Web Mining."
        ),
        "url": _scholar_url("Christian Vargas", "information retrieval recuperacion informacion"),
        "fuente": "Demo - Perfiles Docentes PE",
    },
    {
        "nombre": "Roberto Alarcon Loayza",
        "titulo": "Docente — Bases de Datos y Modelamiento",
        "empresa": "UPC — Universidad Peruana de Ciencias Aplicadas",
        "area": "Tecnología de la Información",
        "ubicacion": "Monterrico, Lima",
        "anios": 11,
        "especialidad": "SQL, PostgreSQL, MongoDB, Modelamiento Relacional, Data Modeling",
        "descripcion": (
            "Docente de Bases de Datos Relacionales y NoSQL. "
            "Experiencia profesional como DBA senior en sector financiero peruano. "
            "Certificación Oracle Database. Proyectos con Banco Pichincha y Claro Perú."
        ),
        "url": _scholar_url("Roberto Alarcon", "bases datos modelamiento"),
        "fuente": "Demo - Perfiles Docentes PE",
    },
    {
        "nombre": "David Mauricio Saenz",
        "titulo": "Docente Investigador — Redes Neuronales y Deep Learning",
        "empresa": "UNMSM — Universidad Nacional Mayor de San Marcos",
        "area": "Educación Superior",
        "ubicacion": "Cercado de Lima",
        "anios": 13,
        "especialidad": "Redes Neuronales, Deep Learning, CNN, Optimización, TensorFlow",
        "descripcion": (
            "Doctor en Ingeniería de Sistemas. Investigador en redes neuronales "
            "y metaheurísticas de optimización. Publicaciones en revistas Q1 de IEEE. "
            "Docente de posgrado en IA y Deep Learning. Proyectos con CONCYTEC."
        ),
        "url": _scholar_url("David Mauricio", "deep learning redes neuronales UNMSM"),
        "fuente": "Demo - Perfiles Docentes PE",
    },
    {
        "nombre": "Hector Zevallos Solis",
        "titulo": "Docente — Matemáticas y Estadística para Ciencia de Datos",
        "empresa": "UNMSM — Universidad Nacional Mayor de San Marcos",
        "area": "Educación Superior",
        "ubicacion": "Cercado de Lima",
        "anios": 8,
        "especialidad": "Álgebra Lineal, Probabilidades, Estadística Matemática, R, Python",
        "descripcion": (
            "Magíster en Matemática Aplicada. Docente de fundamentos matemáticos "
            "para Ciencia de Datos: Álgebra Lineal, Cálculo, Probabilidad y Estadística. "
            "Uso de Python y R para enseñanza computacional de matemáticas."
        ),
        "url": _scholar_url("Hector Zevallos", "estadistica matematica ciencia datos"),
        "fuente": "Demo - Perfiles Docentes PE",
    },
    {
        "nombre": "Marco Camargo Pachas",
        "titulo": "Docente — Ingeniería de Software y DevOps",
        "empresa": "UNI — Universidad Nacional de Ingeniería",
        "area": "Tecnología de la Información",
        "ubicacion": "Rímac, Lima",
        "anios": 5,
        "especialidad": "Ingeniería de Software, Scrum, DevOps, Git, Arquitectura",
        "descripcion": (
            "Investigador en Ingeniería de Software y metodologías ágiles. "
            "Docente de Pruebas de Software y DevOps. "
            "Certificaciones PMP, Scrum Master. "
            "Experiencia en proyectos gubernamentales peruanos."
        ),
        "url": _scholar_url("Marco Camargo", "ingenieria software devops"),
        "fuente": "Demo - Perfiles Docentes PE",
    },
    {
        "nombre": "Yudith Cardinale Villalobos",
        "titulo": "Docente Investigadora — Ciberseguridad y Redes",
        "empresa": "UPN — Universidad Privada del Norte",
        "area": "Tecnología de la Información",
        "ubicacion": "Los Olivos, Lima",
        "anios": 9,
        "especialidad": "Ciberseguridad, Seguridad de Información, Redes, ISO 27001",
        "descripcion": (
            "Doctora en Ciencias de la Computación. Investigadora en seguridad "
            "de la información y computación distribuida. Publicaciones en IEEE. "
            "Docente de Seguridad Informática y Hacking Ético."
        ),
        "url": _scholar_url("Yudith Cardinale", "ciberseguridad seguridad informacion"),
        "fuente": "Demo - Perfiles Docentes PE",
    },
    {
        "nombre": "Juan Gutierrez Soto",
        "titulo": "Docente Investigador — Cloud Computing y Microservicios",
        "empresa": "UTEC — Universidad de Ingeniería y Tecnología",
        "area": "Tecnología de la Información",
        "ubicacion": "Barranco, Lima",
        "anios": 7,
        "especialidad": "Cloud Computing, AWS, Docker, Kubernetes, Microservicios",
        "descripcion": (
            "Investigador en Computación en la Nube y arquitecturas distribuidas. "
            "Certificación AWS Solutions Architect. Docente de Cloud Computing. "
            "Proyectos con empresas tech peruanas. Python, Go, Kubernetes."
        ),
        "url": _scholar_url("Juan Gutierrez", "cloud computing microservicios AWS"),
        "fuente": "Demo - Perfiles Docentes PE",
    },
    {
        "nombre": "Nelly Condori Fernandez",
        "titulo": "Docente Investigadora — Minería de Datos y Recomendación",
        "empresa": "UPC — Universidad Peruana de Ciencias Aplicadas",
        "area": "Educación Superior",
        "ubicacion": "Monterrico, Lima",
        "anios": 11,
        "especialidad": "Data Mining, Sistemas de Recomendación, Machine Learning, Python",
        "descripcion": (
            "Doctora en Computer Science. Investigadora en minería de datos "
            "y sistemas de recomendación. Publicaciones en IEEE y ACM. "
            "Python, scikit-learn, Spark MLlib. Docente de posgrado en Data Science."
        ),
        "url": _scholar_url("Nelly Condori", "data mining recommendation systems machine learning"),
        "fuente": "Demo - Perfiles Docentes PE",
    },
]


def _extraer_texto_perfil(html_perfil):
    """Extrae todo el texto relevante de la página de perfil individual."""
    if not html_perfil:
        return ""
    soup = BeautifulSoup(html_perfil, "html.parser")
    for tag in soup(["script", "style", "header", "footer"]):
        tag.decompose()
    return " ".join(soup.get_text(" ", strip=True).split())


def scrape_mock_web(termino, url_remota=None, callback=None, seguir_perfiles=True):
    """Scrapea el portal Bumeran mock desde URL remota (Vercel)."""
    if not url_remota:
        return []

    base_url = url_remota.rstrip("/")
    if callback:
        callback(f"[Mock Web] Conectando a {base_url}/")

    try:
        resp = requests.get(base_url + "/", headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        html_index = resp.text
    except Exception as e:
        if callback:
            callback(f"[Mock Web] Error: {e}")
        return []

    try:
        soup = BeautifulSoup(html_index, "html.parser")
        perfiles = []

        cards = soup.find_all("article", class_="candidato-card")
        if not cards:
            cards = soup.find_all("div", class_="docente-card")

        if callback:
            callback(f"[Mock Web] {len(cards)} candidatos encontrados.")

        for i, card in enumerate(cards, 1):
            data = card.attrs
            nombre    = data.get("data-nombre", "")
            titulo    = data.get("data-titulo", "")
            area      = data.get("data-area", "")
            nivel     = data.get("data-nivel", "")
            ubicacion = data.get("data-ubicacion", "Perú")
            anios     = data.get("data-experiencia", "")
            skills    = data.get("data-skills", "")
            desc_idx  = data.get("data-descripcion", "")
            empresa   = data.get("data-empresa", area)
            salario   = data.get("data-pretension", "")
            card_id   = data.get("data-id", str(i))

            cv_link    = data.get("data-cv", "")
            perfil_link = data.get("data-perfil", "")
            if cv_link and not cv_link.startswith("http"):
                cv_link = base_url + "/" + cv_link.lstrip("/")
            if perfil_link and not perfil_link.startswith("http"):
                perfil_link = base_url + "/" + perfil_link.lstrip("/")

            texto_perfil = ""
            if seguir_perfiles and perfil_link:
                try:
                    r2 = requests.get(perfil_link, headers=_HEADERS, timeout=10)
                    r2.raise_for_status()
                    texto_perfil = _extraer_texto_perfil(r2.text)
                    if callback and i % 10 == 0:
                        callback(f"[Mock Web] {i}/{len(cards)} perfiles procesados")
                except Exception:
                    pass

            descripcion = " ".join(filter(None, [
                titulo, area, nivel,
                f"{anios} años de experiencia" if anios else "",
                skills, desc_idx, texto_perfil,
            ]))
            descripcion = " ".join(descripcion.split())

            perfiles.append({
                "nombre":       nombre or f"Candidato {i}",
                "titulo":       titulo or area,
                "empresa":      empresa,
                "area":         area,
                "ubicacion":    ubicacion,
                "especialidad": titulo,
                "descripcion":  descripcion,
                "skills":       skills,
                "nivel":        nivel,
                "anios":        anios,
                "salario_y_tipo": salario,
                "url":          cv_link or perfil_link,
                "url_perfil":   perfil_link,
                "fuente":       "Portal Candidatos Bumeran (mock)",
            })

        if callback:
            callback(f"[Mock Web] {len(perfiles)} candidatos extraídos.")

        if termino.strip():
            termino_lower = termino.lower()
            palabras = [p for p in termino_lower.split() if len(p) > 2]
            for p in perfiles:
                texto_cmp = " ".join([
                    p.get("nombre", ""), p.get("titulo", ""),
                    p.get("skills", ""), p.get("descripcion", ""),
                ]).lower()
                p["relevancia_termino"] = sum(1 for w in palabras if w in texto_cmp)
            perfiles.sort(key=lambda x: x.get("relevancia_termino", 0), reverse=True)

        return perfiles

    except Exception as e:
        if callback:
            callback(f"[Mock Web] Error parseando: {e}")
        return []


def buscar_perfiles_demo(termino, cantidad=15, callback=None):
    """Filtra el dataset demo de perfiles docentes por relevancia al término."""
    if callback:
        callback(f"[Dataset Demo] Filtrando para: '{termino}'")

    termino_lower = termino.lower()
    palabras = [p for p in termino_lower.split() if len(p) > 2]

    puntuados = []
    for perfil in PERFILES_DEMO:
        texto = " ".join([
            perfil.get("titulo", ""),
            perfil.get("especialidad", ""),
            perfil.get("descripcion", ""),
        ]).lower()
        score = sum(1 for p in palabras if p in texto)
        copia = perfil.copy()
        copia["relevancia_termino"] = score
        puntuados.append(copia)

    puntuados.sort(key=lambda x: x["relevancia_termino"], reverse=True)
    resultado = puntuados[:cantidad]

    if callback:
        callback(f"[Dataset Demo] {len(resultado)} perfiles listos")
    return resultado


def scrape_ofertas(termino, fuentes=None, max_paginas=2, callback=None, url_mockweb=None):
    """Scraping unificado de perfiles de docentes e investigadores."""
    if fuentes is None:
        fuentes = ["demo"]

    todos = []

    if "mockweb" in fuentes and url_mockweb:
        perfiles = scrape_mock_web(termino, url_remota=url_mockweb, callback=callback)
        todos.extend(perfiles)

    if "demo" in fuentes:
        perfiles = buscar_perfiles_demo(termino, callback=callback)
        todos.extend(perfiles)

    # Fallback: si scrapers web no devuelven nada, usar demo
    sin_demo = [p for p in todos if p.get("fuente") != "Demo - Perfiles Docentes PE"]
    if "mockweb" in fuentes and not sin_demo:
        if callback:
            callback("[Fallback] Mock web sin resultados → usando dataset demo")
        todos.extend(buscar_perfiles_demo(termino, callback=callback))

    if callback:
        callback(f"TOTAL: {len(todos)} perfiles encontrados")
    return todos


def ofertas_a_texto_corpus(perfiles):
    """Convierte perfiles a texto plano para indexar en el SRI."""
    textos = []
    for p in perfiles:
        partes = [
            p.get("nombre", ""),
            p.get("empresa", ""),
            p.get("ubicacion", ""),
            p.get("especialidad", ""),
            p.get("descripcion", ""),
        ]
        texto = " ".join(parte for parte in partes if parte).strip()
        if texto:
            textos.append(texto)
    return textos


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE LA APLICACIÓN STREAMLIT
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Sistema de Recuperación de Información",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
h1 { color: #16b4a8; }
.stButton>button { background-color: #16b4a8; color: white; border-radius: 5px; }
.modulo-box {
    background: #f8f9fa; padding: 15px; border-radius: 8px;
    border-left: 4px solid #16b4a8; margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ── Barra lateral ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 SRI Docente")
    st.caption("Arquitectura de 4 Módulos")
    st.divider()
    st.markdown("""
    **Módulos del Sistema:**
    1. Interfaz de Usuario y Filtros
    2. Procesamiento de Texto
    3. Indexación (Índice Invertido)
    4. Coincidencia y Ranking
    """)
    st.divider()
    st.write("📍 **Fuente de datos:**")
    st.code("https://bumeran.vercel.app/")

st.title("Sistema de Recuperación de Información (SRI)")
st.write("Explora la anatomía completa del motor de búsqueda paso a paso.")

# Inicializar estado de sesión
if "perfiles" not in st.session_state:
    st.session_state.perfiles = []
    st.session_state.textos = []
    st.session_state.sistema_ri = None

# Pestañas principales
tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Búsqueda y Filtros",
    "2️⃣ Motor Lingüístico",
    "3️⃣ Indexación",
    "4️⃣ Ranking y Clasificación"
])

# ──────────────────────────────────────────────────────────────────────────────
# PESTAÑA 1: BÚSQUEDA Y FILTROS
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown(
        "<div class='modulo-box'><strong>1. Interfaz de Usuario y Entrada (Lo que tú ves):</strong> "
        "Caja de búsqueda flexible con filtros de facetas para segmentar resultados.</div>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        termino = st.text_input(
            "¿Qué perfil estás buscando? (Soporta lenguaje natural y frases exactas)",
            value="profesores de ciencia de datos con especialidad en minería de datos",
            help="El sistema limpiará tu consulta automáticamente en el siguiente módulo."
        )
    with col2:
        st.write("")
        st.write("")
        btn_buscar = st.button("🔍 Extraer y Analizar", use_container_width=True)

    st.subheader("Filtros de Facetas (Refinamiento)")
    f1, f2, f3 = st.columns(3)
    filtro_area = f1.selectbox(
        "Área / Sector",
        ["Todas", "Educación Superior", "Tecnología de la Información", "Recursos Humanos", "Finanzas"]
    )
    filtro_exp = f2.slider("Experiencia Mínima (Años)", 0, 15, 0)
    filtro_salario = f3.selectbox(
        "Expectativa Salarial",
        ["Cualquiera", "Menos de S/3000", "S/3000 - S/6000", "Más de S/6000"]
    )

    if btn_buscar:
        if not termino.strip():
            st.warning("Por favor, ingresa los criterios de búsqueda.")
        else:
            with st.spinner("Extrayendo documentos desde la fuente (Scraping)..."):
                perfiles_extraidos = scrape_ofertas(
                    termino,
                    fuentes=["mockweb", "demo"],
                    max_paginas=10,
                    url_mockweb="https://bumeran.vercel.app/"
                )

            if perfiles_extraidos:
                perfiles_filtrados = []
                for p in perfiles_extraidos:
                    try:
                        a = float(p.get("anios", 0) or 0)
                    except Exception:
                        a = 0
                    if filtro_area != "Todas" and p.get("area") != filtro_area:
                        continue
                    if a < filtro_exp:
                        continue
                    perfiles_filtrados.append(p)

                st.session_state.perfiles = perfiles_filtrados
                st.session_state.textos = ofertas_a_texto_corpus(perfiles_filtrados)

                nombres = [p.get("nombre", f"Candidato_{i}") for i, p in enumerate(perfiles_filtrados)]
                st.session_state.sistema_ri = SistemaRI(
                    st.session_state.textos,
                    nombres,
                    [len(t.split()) for t in st.session_state.textos]
                )

                st.success(
                    f"✅ Se extrajeron y filtraron {len(perfiles_filtrados)} perfiles. "
                    "¡Ve a la pestaña 2 para ver cómo se procesan!"
                )

                df_res = pd.DataFrame(perfiles_filtrados)
                if not df_res.empty:
                    cols_mostrar = [c for c in ["nombre", "area", "anios", "empresa", "especialidad"] if c in df_res.columns]
                    st.dataframe(df_res[cols_mostrar], use_container_width=True, hide_index=True)
            else:
                st.warning("No se encontraron perfiles. Intenta con otro término.")

# ──────────────────────────────────────────────────────────────────────────────
# PESTAÑA 2: PROCESAMIENTO DE TEXTO
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown(
        "<div class='modulo-box'><strong>2. Procesamiento de Texto (El motor lingüístico):</strong> "
        "Limpieza de palabras mediante tokenización, eliminación de stopwords, "
        "lematización (stemming) y normalización.</div>",
        unsafe_allow_html=True
    )

    if st.session_state.perfiles:
        st.subheader("Análisis de la Consulta de Búsqueda")
        st.info(f"Consulta original: **\"{termino}\"**")

        t_tokens = tokenizar(termino)
        t_sin_sw = eliminar_stopwords(t_tokens)
        t_stem   = aplicar_stemming(t_sin_sw)

        c1, c2, c3 = st.columns(3)
        c1.markdown("**1. Tokenización y Normalización**")
        c1.caption("Separa en palabras y elimina mayúsculas/acentos.")
        c1.code(" | ".join(t_tokens))

        c2.markdown("**2. Eliminación de Stopwords**")
        c2.caption("Ignora preposiciones y conectores (de, con, en).")
        c2.code(" | ".join(t_sin_sw))

        c3.markdown("**3. Lematización / Stemming**")
        c3.caption("Reduce la palabra a su raíz fonética.")
        c3.code(" | ".join(t_stem))

        st.divider()
        st.subheader("Procesamiento de los Documentos (CVs)")
        st.write(
            "El mismo proceso se aplica a los textos largos de los currículums "
            "extraídos para poder compararlos matemáticamente."
        )
        if st.session_state.textos:
            muestra = st.session_state.textos[0]
            st.text_area("Ejemplo de CV original (Fragmento):", muestra[:300] + "...", height=100)
            m_tokens = tokenizar(muestra)
            m_sin_sw = eliminar_stopwords(m_tokens)
            m_stem   = aplicar_stemming(m_sin_sw)
            st.code(" | ".join(m_stem[:25]) + " ... (continúa)")
    else:
        st.warning("Realiza una búsqueda en la pestaña 1 primero.")

# ──────────────────────────────────────────────────────────────────────────────
# PESTAÑA 3: INDEXACIÓN
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown(
        "<div class='modulo-box'><strong>3. Indexación (El almacenamiento):</strong> "
        "El Índice Invertido guarda una lista de todas las palabras únicas y apunta "
        "exactamente en qué documentos aparecen para acelerar la búsqueda.</div>",
        unsafe_allow_html=True
    )

    if st.session_state.sistema_ri:
        motor_tfidf = st.session_state.sistema_ri.motor_tfidf

        c1, c2, _ = st.columns(3)
        c1.metric("Documentos Indexados", len(st.session_state.textos))
        c2.metric("Tamaño del Vocabulario (Términos únicos)", motor_tfidf.vocabulario_size)

        st.subheader("Estructura del Índice Invertido")
        st.write("Representación del índice invertido generado en memoria:")

        vocab_muestra = list(motor_tfidf.vocabulario.keys())[:20]
        indice_preview = []
        for word in vocab_muestra:
            idx = motor_tfidf.vocabulario[word]
            docs_aparece = [
                f"Doc_{i+1}"
                for i, vector in enumerate(motor_tfidf.matriz_tfidf)
                if vector[idx] > 0
            ]
            indice_preview.append({
                "Término (Raíz)": word,
                "ID Término": idx,
                "Aparece en Documentos": ", ".join(docs_aparece[:5]) + ("..." if len(docs_aparece) > 5 else "")
            })

        st.dataframe(pd.DataFrame(indice_preview), use_container_width=True)
    else:
        st.warning("Realiza una búsqueda en la pestaña 1 primero.")

# ──────────────────────────────────────────────────────────────────────────────
# PESTAÑA 4: COINCIDENCIA Y CLASIFICACIÓN
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown(
        "<div class='modulo-box'><strong>4. Coincidencia y Clasificación (Ranking):</strong> "
        "Algoritmos de relevancia (TF-IDF y BM25) calculan qué tan bien coincide tu "
        "consulta con los documentos y los ordenan por puntuación.</div>",
        unsafe_allow_html=True
    )

    if st.session_state.sistema_ri:
        resultado_sc = st.session_state.sistema_ri.buscar_comparativo(
            termino, top_k=min(10, len(st.session_state.textos))
        )

        t_radar, t_bm25, t_tfidf = st.tabs([
            "🕸️ Radar de Contratación",
            "🏆 Algoritmo BM25",
            "🏆 Algoritmo TF-IDF"
        ])

        with t_radar:
            st.info(
                "Visualización multicriterio de los 3 mejores perfiles encontrados "
                "cruzando la relevancia algorítmica con las reglas de negocio (C1 a C5)."
            )

            def graficar_radar(perfiles_list, res_bm25, res_tfidf, top_n=3):
                top_nombres = [r["archivo"] for r in res_bm25["resultados"][:top_n]]
                fig = go.Figure()
                categorias = [
                    'Formación Académica', 'Experiencia Docente',
                    'Experiencia Profesional', 'Centro de Labores', 'Investigación'
                ]

                for i, nombre in enumerate(top_nombres):
                    p = next((x for x in perfiles_list if x.get("nombre") == nombre), None)
                    if not p:
                        continue

                    txt = (
                        str(p.get("descripcion", "")).lower() + " " +
                        str(p.get("titulo", "")).lower()
                    )

                    if "doctor" in txt or "phd" in txt:
                        val_formacion = 10
                    elif "maestr" in txt or "master" in txt or "magister" in txt:
                        val_formacion = 8
                    elif "bachiller" in txt or "licenciad" in txt:
                        val_formacion = 5
                    else:
                        val_formacion = 3

                    try:
                        anios = float(p.get("anios", 0) or 0)
                    except Exception:
                        anios = 0

                    es_docente = "profesor" in txt or "docente" in txt or p.get("area") == "Educación Superior"
                    val_docencia = min((anios / 5.0) * 10, 10) if es_docente else min((anios / 10.0) * 4, 4)

                    es_corp = (
                        p.get("area") == "Tecnología de la Información" or
                        "gerente" in txt or "jefe" in txt or "senior" in txt
                    )
                    val_profesional = min((anios / 7.0) * 10, 10) if es_corp else min((anios / 10.0) * 5, 5)
                    if "gerente" in txt or "director" in txt or "lead" in txt:
                        val_profesional = min(val_profesional + 2, 10)

                    empresa = str(p.get("empresa", "")).upper()
                    top_empresas = ["GOOGLE", "AWS", "PUCP", "BCP", "INTERBANK", "UPC", "GLOBANT", "TELEFÓNICA"]
                    val_empresa = 10 if empresa in top_empresas else (7 if empresa and empresa != "N/A" else 4)

                    if "scopus" in txt or "wos" in txt or "paper" in txt or "q1" in txt:
                        val_investigacion = 10
                    elif "investigación" in txt or "revista" in txt or "publicación" in txt:
                        val_investigacion = 7
                    elif "proyecto" in txt or "innovación" in txt:
                        val_investigacion = 4
                    else:
                        val_investigacion = 1

                    vals = [val_formacion, val_docencia, val_profesional, val_empresa, val_investigacion]

                    if i == 0:
                        nombre_leyenda = f"GANADOR: {nombre}"
                        color = "#16b4a8"
                        ancho = 3
                        dash = "solid"
                        fillcolor = "rgba(22, 180, 168, 0.4)"
                    else:
                        nombre_leyenda = f"Top {i+1}: {nombre}"
                        color = "#ff7f0e" if i == 1 else "#2ca02c"
                        ancho = 2
                        dash = "dot"
                        fillcolor = "rgba(0,0,0,0)"

                    fig.add_trace(go.Scatterpolar(
                        r=vals + [vals[0]],
                        theta=categorias + [categorias[0]],
                        fill="toself",
                        fillcolor=fillcolor,
                        name=nombre_leyenda,
                        line=dict(color=color, width=ancho, dash=dash)
                    ))

                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                    showlegend=True,
                    height=450,
                    title="Comparador de Radares: Ganador vs Resto"
                )
                return fig

            if resultado_sc["bm25"]["resultados"]:
                fig_radar = graficar_radar(
                    st.session_state.perfiles,
                    resultado_sc["bm25"],
                    resultado_sc["tfidf"],
                    top_n=3
                )
                st.plotly_chart(fig_radar, use_container_width=True)

        with t_bm25:
            st.write("Resultados usando el modelo probabilístico BM25 (Mejor manejo de longitud de documentos).")
            df_bm25 = (
                pd.DataFrame(resultado_sc["bm25"]["resultados"])
                .rename(columns={"archivo": "Candidato", "score": "Puntaje BM25", "rank": "Puesto"})
            )
            st.dataframe(df_bm25[["Puesto", "Candidato", "Puntaje BM25"]], use_container_width=True, hide_index=True)

        with t_tfidf:
            st.write("Resultados usando modelo vectorial TF-IDF (Frecuencia de término - Frecuencia inversa).")
            df_tfidf = (
                pd.DataFrame(resultado_sc["tfidf"]["resultados"])
                .rename(columns={"archivo": "Candidato", "score": "Puntaje TF-IDF", "rank": "Puesto"})
            )
            st.dataframe(df_tfidf[["Puesto", "Candidato", "Puntaje TF-IDF"]], use_container_width=True, hide_index=True)
    else:
        st.warning("Realiza una búsqueda en la pestaña 1 primero.")
