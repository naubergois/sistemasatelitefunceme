import os
import json
import logging
import requests
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import traceback
import io

from datetime import datetime
from netCDF4 import Dataset
from shapely.geometry import Point
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import TypedDict, List, Annotated
import operator

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from geopy.distance import geodesic

# Import the existing download tool
from queimadas_goes16 import download_goes_data

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wildfire_graph")

# --- STATIC DATA: FIRE STATIONS (CBMCE) ---
# Coordinates approximated for major battalions
CBMCE_STATIONS = {
    "1ª Cia/1º BBM - Fortaleza (Jacarecanga)": (-3.722, -38.538),
    "2ª Cia/1º BBM - Fortaleza (Mucuripe)": (-3.725, -38.480),
    "1ª Cia/3º BBM - Sobral": (-3.688, -40.349),
    "2ª Cia/3º BBM - Tianguá": (-3.731, -40.993),
    "1ª Cia/4º BBM - Iguatu": (-6.368, -39.303),
    "1ª Cia/5º BBM - Juazeiro do Norte": (-7.221, -39.328),
    "2ª Cia/5º BBM - Crato": (-7.234, -39.412),
    "1ª Cia/BCIF - Quixadá": (-4.970, -39.016),
    "2ª Cia/BCIF - Quixeramobim": (-5.197, -39.294),
    "3ª Cia/BCIF - Canindé": (-4.358, -39.313),
    "1ª Cia/2º BBM - Maracanaú": (-3.876, -38.626),
    "2ª Cia/2º BBM - Horizonte": (-4.095, -38.490),
    "3ª Cia/2º BBM - Caucaia": (-3.737, -38.653),
    "2ª Cia/4º BBM - Limoeiro do Norte": (-5.147, -38.098),
    "3ª Cia/4º BBM - Aracati": (-4.561, -37.767),
    "2ª Cia/3º BBM - Crateús": (-5.176, -40.668),
    "3ª Cia/3º BBM - Tauá": (-6.002, -40.293),
    "4ª Cia/3º BBM - Itapipoca": (-3.494, -39.586)
}

# --- STATE DEFINITION ---
class WildfireState(TypedDict):
    date_query: str
    nc_paths: dict  # {"b07": path, "b13": path}
    raw_anomalies: int
    confirmed_anomalies: int
    fire_coordinates: List[tuple] # [(lat, lon, temp), ...]
    enriched_data: List[dict] # [{lat, lon, city, region, station, dist_km}, ...]
    map_image_path: str
    analyst_report: str
    error: str

# --- NODES ---

def fetch_data_node(state: WildfireState):
    """Downloads GOES-16 data for the requested date."""
    query = state["date_query"]
    logger.info(f"Fetching data for: {query}")
    try:
        result_json = download_goes_data.invoke({"query": query})
        if "Erro" in result_json and not result_json.strip().startswith("{"):
            return {"error": result_json}
        
        paths = json.loads(result_json)
        return {"nc_paths": paths}
    except Exception as e:
        return {"error": str(e)}

def refine_detection_node(state: WildfireState):
    """Runs K-Means filters for high temperature."""
    if state.get("error"): return state

    paths = state["nc_paths"]
    try:
        nc07 = Dataset(paths['b07']); nc13 = Dataset(paths['b13'])
        t07 = nc07.variables['CMI'][:]; t13 = nc13.variables['CMI'][:]
        
        # Georeferencing
        proj_var = nc07.variables['goes_imager_projection']
        h = proj_var.perspective_point_height
        lon_0 = proj_var.longitude_of_projection_origin
        p = pyproj.Proj(proj='geos', h=h, lon_0=lon_0, sweep='x')
        x = nc07.variables['x'][:] * h; y = nc07.variables['y'][:] * h
        xx, yy = np.meshgrid(x, y)
        lons, lats = p(xx, yy, inverse=True)
        
        # Ceará Mask
        mask_ceara = (lats >= -7.9) & (lats <= -2.7) & (lons >= -41.5) & (lons <= -37.1)
        if not np.any(mask_ceara): return {"error": "No pixels inside Ceará bounding box."}

        # Feature Extraction
        features_b07 = t07[mask_ceara].flatten()
        features_b13 = t13[mask_ceara].flatten()
        valid = ~np.isnan(features_b07) & ~np.isnan(features_b13)
        X = np.column_stack((features_b07[valid], features_b13[valid], (features_b07 - features_b13)[valid]))
        
        # K-Means
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        labels = kmeans.fit_predict(StandardScaler().fit_transform(X))
        
        # Fire Cluster
        means = [np.mean(X[labels == i, 0]) for i in range(4)]
        fire_cluster_id = np.argmax(means)
        raw_count = np.sum(labels == fire_cluster_id)
        
        # Strict Filter (>315K)
        fire_indices = (labels == fire_cluster_id)
        temps_in_cluster = X[fire_indices, 0]
        diffs_in_cluster = X[fire_indices, 2]
        strict_mask = (temps_in_cluster > 315) & (diffs_in_cluster > 5)
        confirmed_count = np.sum(strict_mask)
        
        # Coordinates
        ceara_lats = lats[mask_ceara].flatten()[valid][fire_indices]
        ceara_lons = lons[mask_ceara].flatten()[valid][fire_indices]
        
        final_lats = ceara_lats[strict_mask]
        final_lons = ceara_lons[strict_mask]
        final_temps = temps_in_cluster[strict_mask]
        
        # Sample for next steps (Enrich max 10 points to avoid strict API limits/delays)
        # We prioritize the HOTTEST ones.
        sorted_indices = np.argsort(final_temps)[::-1] # Descending temp
        
        coords = []
        limit = min(len(final_lats), 10) # Enriched sample limit
        
        for i in range(limit):
             idx = sorted_indices[i]
             coords.append((float(final_lats[idx]), float(final_lons[idx]), float(final_temps[idx])))
            
        nc07.close(); nc13.close()
        
        return {
            "raw_anomalies": int(raw_count),
            "confirmed_anomalies": int(confirmed_count),
            "fire_coordinates": coords # Only pass the hottest sample for enrichment
        }
        
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

def enrich_location_node(state: WildfireState):
    """Identifies municipalities and nearest fire stations."""
    if state.get("error") or not state.get("fire_coordinates"):
        return {"enriched_data": []}
    
    try:
        # Download Ceará Municipalities (IBGE) - GeoJSON
        url = "https://servicodados.ibge.gov.br/api/v3/malhas/estados/CE?formato=application/vnd.geo+json&qualidade=minima&divisao=municipio"
        
        # Download and Cache in memory variable for optimization (in graph execution context, we re-download, ideally cache globally)
        # For this script we download.
        response = requests.get(url)
        response.raise_for_status()
        gdf_cities = gpd.read_file(io.BytesIO(response.content))
        
        enriched_list = []
        
        for (lat, lon, temp) in state["fire_coordinates"]:
            point = Point(lon, lat)
            
            # 1. Identify City
            # Optimized: Check within bounding box first or just iterate. 
            # Since number of cities is small (184), simple iteration is fine or sjoin if we made a GDF of points.
            # Let's use simple iteration for the few points we have.
            
            city_name = "Desconhecido (Zona Rural/Mar)"
            region_name = "N/A" # Macroregion info would need another shapefile/join, assuming just city for now.
            
            # Check contains
            # GDF index is usually feature index, 'name' might be in columns
            # IBGE usually returns 'codarea' and potentially 'name' if requested or we need to merge.
            # The IBGE GeoJSON often has just geometries. Let's inspect column names dynamically if possible or assume logic.
            # Actually IBGE "malhas" endpoint usually gives just geometry matching iso codes unless "nome" is requested in a specific way?
            # It seems IBGE Malhas V3 returns properties: {codarea: "..."}
            # We might need to fetch names separately or use a different endpoint that includes attributes.
            # Alternative: "https://servicodados.ibge.gov.br/api/v3/malhas/estados/CE?formato=application/vnd.geo+json&qualidade=minima" gives the state.
            # To get names we might need to query the localities API.
            # For simplicity in this demo, if names are missing, we will report "Município ID: X".
            # Update: GeoPandas read_file from that URL usually gets 'codarea'.
            
            match = gdf_cities[gdf_cities.contains(point)]
            if not match.empty:
                # Try to find a name column. If not, use the first column.
                # In IBGE malhas, often we get just the code. We can try to look it up? 
                # Or use a geocoding lib. Reverse geocoding with geopy/Nominatim is an option but rate limited.
                # Let's try Nominatim for the 10 points. It's cleaner.
                
                from geopy.geocoders import Nominatim
                geolocator = Nominatim(user_agent="ceara_wildfire_monitor")
                try:
                    location = geolocator.reverse(f"{lat}, {lon}", exactly_one=True, language='pt')
                    if location:
                        address = location.raw.get('address', {})
                        city_name = address.get('city') or address.get('town') or address.get('village') or "Desconhecido"
                        region_name = address.get('state_district', 'Ceará')
                except:
                    city_name = f"Município (ID {match.iloc[0].values[0]})"
            
            # 2. Nearest Fire Station
            min_dist = float('inf')
            nearest_station = "N/A"
            
            for station, (slat, slon) in CBMCE_STATIONS.items():
                dist = geodesic((lat, lon), (slat, slon)).km
                if dist < min_dist:
                    min_dist = dist
                    nearest_station = station
            
            enriched_list.append({
                "lat": lat, "lon": lon, "temp_k": temp,
                "city": city_name, "region": region_name,
                "station": nearest_station, "dist_km": round(min_dist, 1)
            })
            
        return {"enriched_data": enriched_list}
        
    except Exception as e:
        traceback.print_exc()
        return {"error": f"Enrichment Error: {str(e)}"}

def map_generation_node(state: WildfireState):
    """Generates the map using Geopandas."""
    if state.get("error"): return state

    try:
        url = "https://servicodados.ibge.gov.br/api/v3/malhas/estados/CE?formato=application/vnd.geo+json&qualidade=minima"
        response = requests.get(url)
        gdf_ce = gpd.read_file(io.BytesIO(response.content))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        gdf_ce.plot(ax=ax, color='white', edgecolor='black')
        
        # Plot Fire Points
        coords = state.get("fire_coordinates", [])
        if coords:
            lats = [c[0] for c in coords]
            lons = [c[1] for c in coords]
            ax.scatter(lons, lats, c='red', marker='x', s=50, label='Focos (>315K)')
            
            # Also plot the nearest fire stations for the TOP 3 hottest (to avoid clutter)
            enriched = state.get("enriched_data", [])
            for item in enriched[:3]:
                # Locate station coord
                st_name = item['station']
                if st_name in CBMCE_STATIONS:
                    slat, slon = CBMCE_STATIONS[st_name]
                    ax.scatter(slon, slat, c='blue', marker='o', s=40)
                    # Draw line
                    ax.plot([item['lon'], slon], [item['lat'], slat], 'g--', linewidth=0.5, alpha=0.7)
            
            plt.legend()
            
        ax.set_title(f"Queimadas Ceará + Unidades CBMCE\nFocos Confirmados: {state['confirmed_anomalies']}")
        
        output_path = "mapa_langgraph_ceara.png"
        plt.savefig(output_path)
        plt.close()
        return {"map_image_path": output_path}
    except Exception as e:
        return {"error": f"Map Error: {str(e)}"}

def expert_analyst_node(state: WildfireState):
    """Generates a text report with enriched location data."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.5)

    if state.get("error"):
        # ... (Error handling logic same as before)
        error_msg = state["error"]
        prompt = f"""
        Você é um especialista em monitoramento de queimadas (GOES-16).
        Problema na data: {state.get('date_query')}.
        Erro: "{error_msg}".
        Justifique tecnicamente para o usuário (ex: data futura, delay NOAA).
        """
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"analyst_report": response.content, "error": None}

    # Success
    count = state.get("confirmed_anomalies", 0)
    enriched = state.get("enriched_data", [])
    
    # Format the enriched data for the prompt
    loc_details = ""
    for item in enriched:
        loc_details += f"- {item['city']} ({item['region']}): {item['temp_k']:.1f}K. Base mais próxima: {item['station']} (~{item['dist_km']}km).\n"
    
    prompt = f"""
    Você é um Comandante Estratégico do Corpo de Bombeiros (CBMCE).
    
    SITUAÇÃO ATUAL (Ceará - Satélite GOES-16):
    - Total de Focos Confirmados (>315K): {count}
    
    DETALHAMENTO DOS FOCOS MAIS CRÍTICOS (Top 10):
    {loc_details}
    
    Gere um relatório Operacional e Técnico:
    
    1. **Análise de Impacto e Mobilização**:
       - Cite as cidades atingidas e as macrorregiões.
       - Indique quais batalhões (bases) devem ser acionados prioritariamente pela proximidade.
       - Estime o risco (Baixo/Médio/Crítico) baseado na temperatura e proximidade urbana.
       
    2. **Dados Técnicos**:
       - Liste as coordenadas exatas dos focos críticos para envio de viaturas.
       
    3. **Justificativa Metodológica** (Obrigatório):
       - Satélite GOES-16 (Bandas 07/13).
       - Algoritmo K-Means + Filtro Térmico (>315K).
       - Geocodificação Reversa para identificar municípios.
       - Cálculo Geodésico para roteamento da viatura mais próxima.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"analyst_report": response.content}

# --- GRAPH CONSTRUCTION ---
workflow = StateGraph(WildfireState)

workflow.add_node("fetch_data", fetch_data_node)
workflow.add_node("detect", refine_detection_node)
workflow.add_node("enrich", enrich_location_node) # New Node
workflow.add_node("map", map_generation_node)
workflow.add_node("analyst", expert_analyst_node)

workflow.set_entry_point("fetch_data")

def should_process(state: WildfireState):
    return "analyst" if state.get("error") else "detect"

def should_enrich(state: WildfireState):
    return "analyst" if state.get("error") else "enrich"

def should_map(state: WildfireState):
    return "analyst" if state.get("error") else "map"

workflow.add_conditional_edges("fetch_data", should_process, {"analyst": "analyst", "detect": "detect"})
workflow.add_conditional_edges("detect", should_enrich, {"analyst": "analyst", "enrich": "enrich"})
workflow.add_conditional_edges("enrich", should_map, {"analyst": "analyst", "map": "map"})

workflow.add_edge("map", "analyst")
workflow.add_edge("analyst", END)

app = workflow.compile()

def run_agent(date_query: str):
    return app.invoke({"date_query": date_query})
