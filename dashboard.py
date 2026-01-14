import streamlit as st
import os
import datetime
import traceback
from PIL import Image

st.set_page_config(page_title="Monitor de Queimadas GOES-16 (Expert)", layout="wide")

# Import the new LangGraph agent
try:
    from agent_graph import run_agent
except ImportError as e:
    st.error(f"Erro ao importar agent_graph: {e}")


st.title("🔥 Monitor de Queimadas - Sistema Especialista (LangGraph)")
st.markdown("""
Este painel utiliza um **Agente Especialista (LangGraph)** que orquestra:
1. Coleta de Dados GOES-16.
2. Detecção IA Refinada (K-Means + Filtros Estatísticos).
3. Geração de Mapa Oficial (Geopandas - IBGE).
4. Parecer Técnico Automático (LLM).
""")

# --- SIDEBAR ---
st.sidebar.header("Configuração")

default_date = datetime.date(2025, 4, 5)
selected_date = st.sidebar.date_input("Data da Análise", default_date)
selected_hour = st.sidebar.slider("Hora (UTC)", 0, 23, 15)
query_str = f"{selected_date} {selected_hour:02d}:00"

st.sidebar.markdown("---")
if st.sidebar.button("🚀 Iniciar Análise Especialista"):
    with st.spinner(f"O Agente Especialista está analisando {query_str}..."):
        try:
            # Run the Graph
            result = run_agent(query_str)
            
            if result.get("error"):
                st.error(f"Erro no Agente: {result['error']}")
            else:
                st.success("Análise Finalizada com Sucesso!")
                
                # --- VISUALIZAÇÃO ---
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("Mapa de Situação")
                    map_path = result.get("map_image_path")
                    if map_path and os.path.exists(map_path):
                        img = Image.open(map_path)
                        st.image(img, caption="Focos Confirmados em Mapas Oficiais", use_column_width=True)
                    else:
                        st.warning("Mapa não gerado.")
                        
                with col2:
                    st.subheader("Relatório & Justificativa")
                    report = result.get("analyst_report", "Parecer indisponível.")
                    with st.container(border=True):
                        st.markdown(report)
                    
                    st.divider()
                    st.metric("Anomalias Brutas (Cluster)", result.get("raw_anomalies", 0), help="Total de pixels no cluster mais quente (inclui solo quente)")
                    st.metric("Focos Confirmados (>315K)", result.get("confirmed_anomalies", 0), help="Pixels filtrados com alta probabilidade de fogo")

        except Exception as e:
            st.error(f"Falha Crítica: {str(e)}")
            st.code(traceback.format_exc())

st.sidebar.markdown("---")
st.sidebar.info("Sistema v2.0 - LangGraph + Geopandas")
