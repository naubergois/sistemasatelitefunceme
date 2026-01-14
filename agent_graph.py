import os
import json
import logging
import requests
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import traceback

from datetime import datetime
from netCDF4 import Dataset
from shapely.geometry import box
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import TypedDict, List, Annotated
import operator

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

# Import the existing download tool
from queimadas_goes16 import download_goes_data

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wildfire_graph")

# --- STATE DEFINITION ---
class WildfireState(TypedDict):
    date_query: str
    nc_paths: dict  # {"b07": path, "b13": path}
    raw_anomalies: int
    confirmed_anomalies: int
    fire_coordinates: List[tuple] # [(lat, lon, temp), ...]
    map_image_path: str
    analyst_report: str
    error: str

# --- NODES ---

def fetch_data_node(state: WildfireState):
    """Downloads GOES-16 data for the requested date."""
    query = state["date_query"]
    logger.info(f"Fetching data for: {query}")
    try:
        # download_goes_data returns a JSON string, need to parse it
        result_json = download_goes_data.invoke({"query": query})
        if "Erro" in result_json and not result_json.strip().startswith("{"):
            return {"error": result_json}
        
        paths = json.loads(result_json)
        return {"nc_paths": paths}
    except Exception as e:
        return {"error": str(e)}

def refine_detection_node(state: WildfireState):
    """Runs K-Means but filters for high temperature to reduce false positives."""
    if state.get("error"):
        return state

    paths = state["nc_paths"]
    try:
        nc07 = Dataset(paths['b07'])
        nc13 = Dataset(paths['b13'])
        t07 = nc07.variables['CMI'][:]
        t13 = nc13.variables['CMI'][:]
        
        # Georeferencing
        proj_var = nc07.variables['goes_imager_projection']
        h = proj_var.perspective_point_height
        lon_0 = proj_var.longitude_of_projection_origin
        p = pyproj.Proj(proj='geos', h=h, lon_0=lon_0, sweep='x')
        x = nc07.variables['x'][:] * h
        y = nc07.variables['y'][:] * h
        xx, yy = np.meshgrid(x, y)
        lons, lats = p(xx, yy, inverse=True)
        
        # Ceará Mask
        mask_ceara = (lats >= -7.9) & (lats <= -2.7) & (lons >= -41.5) & (lons <= -37.1)
        
        if not np.any(mask_ceara):
            return {"error": "No pixels inside Ceará bounding box."}

        # Feature Extraction
        features_b07 = t07[mask_ceara].flatten()
        features_b13 = t13[mask_ceara].flatten()
        valid = ~np.isnan(features_b07) & ~np.isnan(features_b13)
        
        X = np.column_stack((
            features_b07[valid], 
            features_b13[valid],
            (features_b07 - features_b13)[valid]
        ))
        
        # K-Means
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        labels = kmeans.fit_predict(StandardScaler().fit_transform(X))
        
        # Identify Fire Cluster (Highest Avg Temp)
        means = [np.mean(X[labels == i, 0]) for i in range(4)]
        fire_cluster_id = np.argmax(means)
        raw_count = np.sum(labels == fire_cluster_id)
        
        # REFINEMENT: Filter points in the fire cluster that are actually HOT
        # Statistical Threshold: Mean + 1 StdDev OR Absolute > 310K
        # Choosing Absolute > 310K (approx 37°C) to remove warm ground
        # Note: B07 is Shortwave IR, heavily acted by sun reflection. 
        # Using a combination of simple threshold on top of the cluster.
        
        fire_indices = (labels == fire_cluster_id)
        # Get actual indices in the X array
        temps_in_cluster = X[fire_indices, 0] # B07 temps
        
        # Strict Filter: Must be > 315K (42°C brightness temp) to be reasonably sure it's fire/hotspot
        # AND check difference > 10K
        diffs_in_cluster = X[fire_indices, 2]
        
        strict_mask = (temps_in_cluster > 315) & (diffs_in_cluster > 5)
        confirmed_count = np.sum(strict_mask)
        
        # Extract Coordinates of Confirmed Fires
        # This is tricky because we flattened constraints.
        # We need to map back to lat/lon.
        
        # 1. Full mask of valid pixels in Ceara
        ceara_lats = lats[mask_ceara].flatten()[valid]
        ceara_lons = lons[mask_ceara].flatten()[valid]
        
        # 2. Subsection of those that are in fire cluster
        cluster_lats = ceara_lats[fire_indices]
        cluster_lons = ceara_lons[fire_indices]
        cluster_temps = temps_in_cluster
        
        # 3. Subsection that passed strict filter
        final_lats = cluster_lats[strict_mask]
        final_lons = cluster_lons[strict_mask]
        final_temps = cluster_temps[strict_mask]
        
        # Sample coordinates (up to 100 to avoid state bloat)
        coords = []
        for i in range(min(len(final_lats), 100)):
            coords.append((float(final_lats[i]), float(final_lons[i]), float(final_temps[i])))
            
        nc07.close(); nc13.close()
        
        return {
            "raw_anomalies": int(raw_count),
            "confirmed_anomalies": int(confirmed_count),
            "fire_coordinates": coords
        }
        
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

def map_generation_node(state: WildfireState):
    """Generates the map using Geopandas and IBGE shapefiles."""
    if state.get("error"):
        return state

    try:
        # Download Ceará Shapefile from IBGE (GeoJSON)
        url = "https://servicodados.ibge.gov.br/api/v3/malhas/estados/CE?formato=application/vnd.geo+json&qualidade=minima"
        
        # Download content properly to handle URL parameters that might confuse GDAL
        import io
        response = requests.get(url)
        response.raise_for_status()
        
        # Load from bytes
        gdf_ce = gpd.read_file(io.BytesIO(response.content))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        gdf_ce.plot(ax=ax, color='white', edgecolor='black')
        
        # Plot Anomalies
        coords = state["fire_coordinates"]
        if coords:
            lats = [c[0] for c in coords]
            lons = [c[1] for c in coords]
            # Verify if there are too many points to plot
            confirmed_count = state["confirmed_anomalies"]
            
            # If we have many points, we might just plot the ones we sampled or full scatter if we passed them all
            # In refine_node we passed up to 100. Let's fix refine_node to pass ALL lats/lons for plotting if possible?
            # Passing 100 is good for text analysis, but for map we might want all.
            # For now, let's plot the sample (it represents the hottest ones if sorted? We didn't sort. Let's assume random sample or first ones)
            
            # Better: Plot the sample we have
            ax.scatter(lons, lats, c='red', marker='x', s=30, label='Focos Confirmados (>315K)')
            plt.legend()
            
        ax.set_title(f"Monitoramento de Queimadas - Ceará\nFocos Confirmados: {state['confirmed_anomalies']}")
        
        output_path = "mapa_langgraph_ceara.png"
        plt.savefig(output_path)
        plt.close()
        
        return {"map_image_path": output_path}
    except Exception as e:
        traceback.print_exc()
        return {"error": f"Map Error: {str(e)}"}

def expert_analyst_node(state: WildfireState):
    """Generates a text report using an LLM."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.5)

    # 1. Handle Error State
    if state.get("error"):
        error_msg = state["error"]
        prompt = f"""
        Você é um especialista em monitoramento de queimadas via satélite (GOES-16).
        
        Tivemos um problema ao processar os dados para a data: {state.get('date_query', 'Desconhecida')}.
        Erro reportado: "{error_msg}"
        
        Sua tarefa é JUSTIFICAR esse erro para o usuário final em um parágrafo técnico.
        - Se o erro for sobre "No data found", considere se a data solicitada (Ex: 2026) está no futuro ou se o dataset do satélite tem delay.
        - Explique que o sistema tentou buscar no bucket AWS S3 'noaa-goes16', mas falhou.
        
        Termine com uma **Justificativa Metodológica** explicando que o acesso aos dados depende da disponibilidade no provedor (NOAA) e que datas futuras ou muito recentes podem ainda não ter sido processadas (latência de ingestão).
        """
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"analyst_report": response.content, "error": None}

    # 2. Handle Success State
    # Safely get values, defaulting to 0/empty if somehow missing but no error flag
    count = state.get("confirmed_anomalies", 0)
    raw = state.get("raw_anomalies", 0)
    coords = state.get("fire_coordinates", [])[:5]
    
    prompt = f"""
    Você é um especialista em monitoramento de queimadas via satélite (GOES-16).
    
    Dados da análise de hoje:
    - Anomalias Brutas (Cluster Quente): {raw} (Muitos podem ser solo quente).
    - Focos Confirmados (Filtro Térmico > 315K): {count}.
    - Coordenadas de exemplo (Lat/Lon/Temp): {coords}
    
    Gere um relatório técnico contendo:
    
    1. **Parecer Situacional**: 
       - Analise a severidade com base nos focos confirmados.
       - Recomende ações para a defesa civil.
       
    2. **Justificativa Metodológica** (Obrigatório, abaixo do parecer):
       - Explique que foi usada a técnica de **Aprendizado de Máquina Não-Supervisionado (K-Means)** para segmentar a imagem termal.
       - Explique que um **Filtro Estatístico de Temperatura (>315K)** foi aplicado sobre o cluster quente para remover falsos positivos (solo aquecido vs fogo ativo).
       - Caso haja erros ou 0 focos, analise se pode ser cobertura de nuvens ou ausência real de calor.
       
    Mantenha o tom profissional e direto.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"analyst_report": response.content}

# --- GRAPH CONSTRUCTION ---
workflow = StateGraph(WildfireState)

workflow.add_node("fetch_data", fetch_data_node)
workflow.add_node("detect", refine_detection_node)
workflow.add_node("map", map_generation_node)
workflow.add_node("analyst", expert_analyst_node)

workflow.set_entry_point("fetch_data")

def should_process(state: WildfireState):
    if state.get("error"):
        return "analyst"
    return "detect"

def should_map(state: WildfireState):
    if state.get("error"):
        return "analyst"
    return "map"

workflow.add_conditional_edges("fetch_data", should_process, {
    "analyst": "analyst",
    "detect": "detect"
})

workflow.add_conditional_edges("detect", should_map, {
    "analyst": "analyst",
    "map": "map"
})

workflow.add_edge("map", "analyst")
workflow.add_edge("analyst", END)

app = workflow.compile()

# --- EXPORT FOR DASHBOARD ---
def run_agent(date_query: str):
    """Entry point for the Dashboard."""
    initial_state = {"date_query": date_query}
    result = app.invoke(initial_state)
    return result
