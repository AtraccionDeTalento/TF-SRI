import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ── Permite importar la carpeta modulos/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modulos.scraper_bumeran import scrape_ofertas, ofertas_a_texto_corpus
from modulos.modelos_ir import SistemaRI
from modulos.preprocesamiento import tokenizar, eliminar_stopwords, aplicar_stemming

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE LA APLICACIÓN
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Sistema de Recuperación de Información",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos simples
st.markdown("""
<style>
h1 { color: #16b4a8; }
.stButton>button { background-color: #16b4a8; color: white; border-radius: 5px; }
.modulo-box { background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #16b4a8; margin-bottom: 20px;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# BARRA LATERAL
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title(" SRI Docente")
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

# Inicializar estado para guardar datos
if "perfiles" not in st.session_state:
    st.session_state.perfiles = []
    st.session_state.textos = []
    st.session_state.sistema_ri = None

# Pestañas principales
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Búsqueda y Filtros", 
    "2. Motor Lingüístico", 
    "3. Indexación", 
    "4. Ranking y Clasificación"
])

# ──────────────────────────────────────────────────────────────────────────────
# PESTAÑA 1: BÚSQUEDA Y FILTROS
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("<div class='modulo-box'><strong>1. Interfaz de Usuario y Entrada (Lo que tú ves):</strong> Caja de búsqueda flexible con filtros de facetas para segmentar resultados.</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        termino = st.text_input(
            "¿Qué perfil estás buscando? (Soporta lenguaje natural y frases exactas)",
            value="profesores de ciencia de datos con especialidad en minería de datos",
            help="El sistema autocompletará y limpiará tu consulta automáticamente en el siguiente módulo."
        )
    with col2:
        st.write("")
        st.write("")
        btn_buscar = st.button("🔍 Extraer y Analizar", use_container_width=True)

    st.subheader("Filtros de Facetas (Refinamiento)")
    f1, f2, f3 = st.columns(3)
    filtro_area = f1.selectbox("Área / Sector", ["Todas", "Educación Superior", "Tecnología de la Información", "Recursos Humanos", "Finanzas"])
    filtro_exp = f2.slider("Experiencia Mínima (Años)", 0, 15, 0)
    filtro_salario = f3.selectbox("Expectativa Salarial", ["Cualquiera", "Menos de S/3000", "S/3000 - S/6000", "Más de S/6000"])

    if btn_buscar:
        if not termino.strip():
            st.warning("Por favor, ingresa los criterios de búsqueda.")
        else:
            with st.spinner("Extrayendo documentos desde la fuente (Scraping)..."):
                perfiles_extraidos = scrape_ofertas(termino, fuentes=["mockweb"], max_paginas=10, url_mockweb="https://bumeran.vercel.app/")
                
            if perfiles_extraidos:
                # Aplicar Filtros de facetas de forma rudimentaria
                perfiles_filtrados = []
                for p in perfiles_extraidos:
                    try: a = float(p.get("anios", 0) or 0)
                    except: a = 0
                    
                    if filtro_area != "Todas" and p.get("area") != filtro_area: continue
                    if a < filtro_exp: continue
                    perfiles_filtrados.append(p)
                
                st.session_state.perfiles = perfiles_filtrados
                st.session_state.textos = ofertas_a_texto_corpus(perfiles_filtrados)
                
                nombres = [f"{p.get('nombre', 'Candidato')}" for p in perfiles_filtrados]
                st.session_state.sistema_ri = SistemaRI(st.session_state.textos, nombres, [len(t.split()) for t in st.session_state.textos])
                
                st.success(f" Se extrajeron y filtraron {len(perfiles_filtrados)} perfiles exitosamente. ¡Ve a la pestaña 2 para ver cómo se procesan!")
                
                df_res = pd.DataFrame(perfiles_filtrados)
                if not df_res.empty:
                    cols_mostrar = [c for c in ["nombre", "area", "anios", "empresa", "especialidad"] if c in df_res.columns]
                    st.dataframe(df_res[cols_mostrar], use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────────────────────
# PESTAÑA 2: PROCESAMIENTO DE TEXTO
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<div class='modulo-box'><strong>2. Procesamiento de Texto (El motor lingüístico):</strong> Limpieza de palabras mediante tokenización, eliminación de stopwords, lematización (stemming) y normalización.</div>", unsafe_allow_html=True)
    
    if st.session_state.perfiles:
        st.subheader("Análisis de la Consulta de Búsqueda")
        st.info(f"Consulta original: **\"{termino}\"**")
        
        t_tokens = tokenizar(termino)
        t_sin_sw = eliminar_stopwords(t_tokens)
        t_stem = aplicar_stemming(t_sin_sw)
        
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
        st.write("El mismo proceso se aplica a los textos largos de los currículums extraídos para poder compararlos matemáticamente.")
        if len(st.session_state.textos) > 0:
            muestra = st.session_state.textos[0]
            st.text_area("Ejemplo de CV original (Fragmento):", muestra[:300] + "...", height=100)
            
            m_tokens = tokenizar(muestra)
            m_sin_sw = eliminar_stopwords(m_tokens)
            m_stem = aplicar_stemming(m_sin_sw)
            st.code(" | ".join(m_stem[:25]) + " ... (continúa)")
    else:
        st.warning("Realiza una búsqueda en la pestaña 1 primero.")

# ──────────────────────────────────────────────────────────────────────────────
# PESTAÑA 3: INDEXACIÓN
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("<div class='modulo-box'><strong>3. Indexación (El almacenamiento):</strong> El Índice Invertido guarda una lista de todas las palabras únicas y apunta exactamente en qué documentos aparecen para acelerar la búsqueda.</div>", unsafe_allow_html=True)
    
    if st.session_state.sistema_ri:
        motor_tfidf = st.session_state.sistema_ri.motor_tfidf
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Documentos Indexados", len(st.session_state.textos))
        c1.metric("Tamaño del Vocabulario (Términos únicos)", motor_tfidf.vocabulario_size)
        
        st.subheader("Estructura del Índice Invertido")
        st.write("A continuación se muestra una representación del índice invertido generado en memoria:")
        
        # Generar una muestra visual del índice invertido
        vocab_muestra = list(motor_tfidf.vocabulario.keys())[:20]
        indice_preview = []
        for word in vocab_muestra:
            idx = motor_tfidf.vocabulario[word]
            # Encontrar en qué docs aparece
            docs_aparece = []
            for i, vector in enumerate(motor_tfidf.matriz_tfidf):
                if vector[idx] > 0:
                    docs_aparece.append(f"Doc_{i+1}")
            indice_preview.append({"Término (Raíz)": word, "ID Término": idx, "Aparece en Documentos": ", ".join(docs_aparece[:5]) + ("..." if len(docs_aparece)>5 else "")})
            
        st.dataframe(pd.DataFrame(indice_preview), use_container_width=True)
    else:
        st.warning("Realiza una búsqueda en la pestaña 1 primero.")

# ──────────────────────────────────────────────────────────────────────────────
# PESTAÑA 4: COINCIDENCIA Y CLASIFICACIÓN
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("<div class='modulo-box'><strong>4. Coincidencia y Clasificación (Ranking):</strong> Algoritmos de relevancia (TF-IDF y BM25) calculan qué tan bien coincide tu consulta con los documentos y los ordenan por puntuación.</div>", unsafe_allow_html=True)
    
    if st.session_state.sistema_ri:
        resultado_sc = st.session_state.sistema_ri.buscar_comparativo(termino, top_k=min(10, len(st.session_state.textos)))
        
        t_radar, t_bm25, t_tfidf = st.tabs([" Radar de Contratación", " Algoritmo BM25", " Algoritmo TF-IDF"])
        
        with t_radar:
            st.info("Visualización multicriterio de los 3 mejores perfiles encontrados cruzando la relevancia algorítmica con las reglas de negocio (C1 a C5).")
            def graficar_radar(perfiles_list, res_bm25, res_tfidf, query, top_n=3):
                top_nombres = [r["archivo"] for r in res_bm25["resultados"][:top_n]]
                map_bm25 = {r["archivo"]: r["score"] for r in res_bm25["resultados"]}
                map_tfidf = {r["archivo"]: r["score"] for r in res_tfidf["resultados"]}
                
                max_b = max(map_bm25.values()) if map_bm25 else 1
                max_t = max(map_tfidf.values()) if map_tfidf else 1
                
                fig = go.Figure()
                categorias = ['Formación Académica', 'Experiencia Docente', 'Experiencia Profesional', 'Centro de Labores', 'Investigación']
                
                for nombre in top_nombres:
                    p = next((x for x in perfiles_list if x.get("nombre") == nombre), None)
                    if not p: continue
                    
                    txt = str(p.get("descripcion", "")).lower() + " " + str(p.get("titulo", "")).lower()
                    
                    if "doctor" in txt or "phd" in txt: val_formacion = 10
                    elif "maestr" in txt or "master" in txt or "magister" in txt: val_formacion = 8
                    elif "bachiller" in txt or "licenciad" in txt: val_formacion = 5
                    else: val_formacion = 3
                    
                    try: anios = float(p.get("anios", 0) or 0)
                    except: anios = 0
                    
                    es_docente = "profesor" in txt or "docente" in txt or p.get("area") == "Educación Superior"
                    val_docencia = min((anios / 5.0) * 10, 10) if es_docente else (min((anios / 10.0) * 4, 4))
                    
                    es_corp = p.get("area") == "Tecnología de la Información" or "gerente" in txt or "jefe" in txt or "senior" in txt
                    val_profesional = min((anios / 7.0) * 10, 10) if es_corp else (min((anios / 10.0) * 5, 5))
                    if "gerente" in txt or "director" in txt or "lead" in txt: val_profesional = min(val_profesional + 2, 10)
                    
                    empresa = str(p.get("empresa", "")).upper()
                    top_empresas = ["GOOGLE", "AWS", "PUCP", "BCP", "INTERBANK", "UPC", "GLOBANT", "TELEFÓNICA"]
                    if empresa in top_empresas: val_empresa = 10
                    elif empresa and empresa != "N/A": val_empresa = 7
                    else: val_empresa = 4
                    
                    if "scopus" in txt or "wos" in txt or "paper" in txt or "q1" in txt: val_investigacion = 10
                    elif "investigación" in txt or "revista" in txt or "publicación" in txt: val_investigacion = 7
                    elif "proyecto" in txt or "innovación" in txt: val_investigacion = 4
                    else: val_investigacion = 1
                    
                    vals = [val_formacion, val_docencia, val_profesional, val_empresa, val_investigacion]
                    fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=categorias + [categorias[0]], fill='toself', name=nombre))
                    
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), showlegend=True, height=450)
                return fig

            if len(resultado_sc["bm25"]["resultados"]) > 0:
                fig_radar = graficar_radar(st.session_state.perfiles, resultado_sc["bm25"], resultado_sc["tfidf"], termino, top_n=3)
                st.plotly_chart(fig_radar, use_container_width=True)
                
        with t_bm25:
            st.write("Resultados usando el modelo probabilístico BM25 (Mejor manejo de longitud de documentos).")
            df_bm25 = pd.DataFrame(resultado_sc["bm25"]["resultados"]).rename(columns={"archivo": "Candidato", "score": "Puntaje BM25", "rank": "Puesto"})
            st.dataframe(df_bm25[["Puesto", "Candidato", "Puntaje BM25"]], use_container_width=True, hide_index=True)
            
        with t_tfidf:
            st.write("Resultados usando modelo vectorial TF-IDF (Frecuencia de término - Frecuencia inversa).")
            df_tfidf = pd.DataFrame(resultado_sc["tfidf"]["resultados"]).rename(columns={"archivo": "Candidato", "score": "Puntaje TF-IDF", "rank": "Puesto"})
            st.dataframe(df_tfidf[["Puesto", "Candidato", "Puntaje TF-IDF"]], use_container_width=True, hide_index=True)
    else:
        st.warning("Realiza una búsqueda en la pestaña 1 primero.")
