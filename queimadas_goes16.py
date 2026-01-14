# ==============================================================================
# SISTEMA DE DETECÇÃO DE QUEIMADAS GOES-16 (VERSÃO V3.0 - UNIVERSAL)
# ==============================================================================

import os
import boto3
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
try:
    from getpass import getpass
except ImportError:
    pass
from dotenv import load_dotenv
from botocore import UNSIGNED
from botocore.config import Config
from netCDF4 import Dataset

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# --- IMPORTS MODERNOS (Agnósticos de Versão) ---
from langchain_openai import ChatOpenAI
# from langchain.agents import create_tool_calling_agent, AgentExecutor # Removed to prevent import errors in new env
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool

# 2. CONFIGURAÇÃO
print("✅ Bibliotecas carregadas.")
if "OPENAI_API_KEY" not in os.environ:
    # Fallback if getpass is problematic in some environments or just to be safe
    try:
        os.environ["OPENAI_API_KEY"] = getpass("🔑 Insira sua OpenAI API Key: ")
    except Exception:
        print("Erro ao ler input. Defina a variável de ambiente OPENAI_API_KEY.")
        exit(1)

# Configuração AWS S3
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
BUCKET_NAME = 'noaa-goes16'

# 3. FERRAMENTAS (TOOLS)

@tool
def download_goes_data(query: str = "") -> str:
    """Baixa dados do GOES-16 (Banda 07 e 13). Aceita data 'YYYY-MM-DD HH' na query ou usa atual."""
    try:
        # Tenta extrair data da query (ex: "2025-04-05 15:00")
        dt = datetime.strptime(query.strip()[:13], "%Y-%m-%d %H")
    except ValueError:
        dt = datetime.now(timezone.utc)

    
    def get_files(dt):
        prefix = f"ABI-L2-CMIPF/{dt.year}/{dt.strftime('%j')}/{dt.hour:02d}/"
        print(f"[*] Buscando bucket: {prefix}...")
        return s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix).get('Contents', [])

    files = get_files(dt)
    if not files:
        print(f"[!] Dados indisponíveis para {dt}. Tentando hora anterior...")
        files = get_files(dt - timedelta(hours=1))
        
    if not files: return "Erro: Dados indisponíveis."

    c07 = sorted([f['Key'] for f in files if "M6C07" in f['Key']])
    c13 = sorted([f['Key'] for f in files if "M6C13" in f['Key']])
    
    if not c07 or not c13: return "Erro: Bandas incompletas."

    if not c07 or not c13: return "Erro: Bandas incompletas."
    
    # Cria diretório de dados se não existir
    os.makedirs("data", exist_ok=True)

    f07, f13 = c07[-1], c13[-1]
    print(f"[*] Baixando: {f07}")
    
    path_b07 = os.path.join("data", "b07.nc")
    path_b13 = os.path.join("data", "b13.nc")
    
    s3.download_file(BUCKET_NAME, f07, path_b07)
    s3.download_file(BUCKET_NAME, f13, path_b13)
    
    return json.dumps({"b07": path_b07, "b13": path_b13})

@tool
def analyze_fire(file_paths_json: str) -> str:
    """Detecta fogo usando IA não-supervisionada (K-Means) focada no Ceará."""
    import pyproj
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    try:
        paths = json.loads(file_paths_json)
        nc07, nc13 = Dataset(paths['b07']), Dataset(paths['b13'])
        t07, t13 = nc07.variables['CMI'][:], nc13.variables['CMI'][:]
        
        # --- GEOREFERENCIAMENTO (Reaproveitado) ---
        proj_var = nc07.variables['goes_imager_projection']
        h = proj_var.perspective_point_height
        lon_0 = proj_var.longitude_of_projection_origin
        p = pyproj.Proj(proj='geos', h=h, lon_0=lon_0, sweep='x')
        x = nc07.variables['x'][:] * h
        y = nc07.variables['y'][:] * h
        xx, yy = np.meshgrid(x, y)
        lons, lats = p(xx, yy, inverse=True)
        
        # --- FILTRO CEARÁ ---
        mask_ceara = (lats >= -7.9) & (lats <= -2.7) & (lons >= -41.5) & (lons <= -37.1)
        
        # Se não houver pontos no Ceará, retorna
        if not np.any(mask_ceara):
            return "Erro: Nenhum pixel válido no Ceará encontrado."

        # --- PREPARAÇÃO PARA IA ---
        # Extrair features apenas dos pixels do Ceará para economizar memória
        # Features: Banda 07 (Térmico), Banda 13 (Limpa), Diferença (7 - 13)
        features_b07 = t07[mask_ceara].flatten()
        features_b13 = t13[mask_ceara].flatten()
        
        # Filtra valores inválidos (Masked/NaN)
        valid_pixels = ~np.isnan(features_b07) & ~np.isnan(features_b13)
        
        if np.sum(valid_pixels) < 100:
            return "Erro: Poucos pixels válidos para clusterização."

        X_valid = np.column_stack((
            features_b07[valid_pixels], 
            features_b13[valid_pixels],
            (features_b07 - features_b13)[valid_pixels]
        ))
        
        # Normalização
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_valid)
        
        # --- CLUSTERIZAÇÃO K-MEANS ---
        # 4 Clusters esperados: Mar/Terra Fria, Terra Quente, Nuvens, FOGO/Anomalia
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # IDENTIFICAR CLUSTER DE FOGO
        # O cluster de fogo deve ter a maior média na Banda 07 (Temperatura de Brilho)
        cluster_means = []
        for i in range(4):
            # Média da feature 0 (Band 07) para este cluster
            mean_temp = np.mean(X_valid[labels == i, 0])
            cluster_means.append((i, mean_temp))
            print(f"Cluster {i}: Média Temp B07 = {mean_temp:.2f} K")
            
        # Ordena clusters por temperatura (o maior é o candidato a fogo)
        cluster_means.sort(key=lambda x: x[1], reverse=True)
        fire_cluster_id = cluster_means[0][0]
        max_temp_avg = cluster_means[0][1]
        
        print(f"🔥 Cluster Candidato a Fogo: {fire_cluster_id} (Temp Média: {max_temp_avg:.2f} K)")
        
        # --- RECONSTRUÇÃO DA IMAGEM ---
        # Cria uma máscara vazia do tamanho da imagem original
        final_fire_mask = np.zeros_like(t07, dtype=bool)
        
        # Mapeia de volta:
        # 1. Cria array de labels do tamanho 'mask_ceara' preenchido com -1
        ceara_labels = np.full(features_b07.shape, -1)
        
        # 2. Preenche os pixels válidos com os labels encontrados
        ceara_labels[valid_pixels] = labels
        
        # 3. Identifica pixels de fogo no recorte plana (apenas onde label == fire_cluster_id)
        is_fire_flat = (ceara_labels == fire_cluster_id)
        
        # 4. Refinamento pós-clustering (Opcional mas recomendado):
        # O cluster mais quente pode ser apenas "terra muito quente".
        # Vamos garantir que também seja quente em valor absoluto (> 312K como sanity check relaxado)
        # Se for puramente não-supervisionado, removemos isso. Vamos manter puro por enquanto,
        # mas adicionamos um aviso se a temperatura média for muito baixa.
        if max_temp_avg < 300:
            print("⚠️ Aviso: O cluster mais quente é < 300K. Provavelmente não há fogo, apenas terra quente.")
        
        # Preenche a máscara original
        final_fire_mask[mask_ceara] = is_fire_flat
        
        y_fire, x_fire = np.where(final_fire_mask)
        count = len(y_fire)
        
        # --- RECORTE (CROP) PARA O CEARÁ ---
        # Encontrar os índices limites da máscara do Ceará
        rows, cols = np.where(mask_ceara)
        if len(rows) > 0:
            y_min, y_max = rows.min(), rows.max()
            x_min, x_max = cols.min(), cols.max()
            
            # Recorta os arrays para plotagem
            t07_crop = t07[y_min:y_max+1, x_min:x_max+1]
            x_crop = x[x_min:x_max+1]
            y_crop = y[y_min:y_max+1]
            
            # Recorta a máscara de fogo também para plotar os pontos
            fire_mask_crop = final_fire_mask[y_min:y_max+1, x_min:x_max+1]
            yf_crop, xf_crop = np.where(fire_mask_crop)
            
            # Cria grids locais para o scatter plot no recorte
            xx_crop, yy_crop = np.meshgrid(x_crop, y_crop)
            
            # --- PLOTAGEM ---
            plt.figure(figsize=(10, 8))
            
            # Plota imagem de fundo térmica (Recortada)
            plt.imshow(t07_crop, cmap='inferno', vmin=280, vmax=340, 
                      extent=[x_crop.min(), x_crop.max(), y_crop.min(), y_crop.max()])
            plt.colorbar(label='Temperatura (K)')
            
            if len(yf_crop) > 0:
                # Plota pontos do cluster de fogo (sobre o grid recortado)
                plt.scatter(xx_crop[fire_mask_crop], yy_crop[fire_mask_crop], c='cyan', marker='x', s=30, label='AI Detected Fire')
                plt.legend()
                
            plt.title(f"Detecção AI (K-Means) - Zoom Ceará\n{count} anomalias | Cluster ID: {fire_cluster_id}")
            plt.axis('off')
            
            plt.savefig("queimadas_ai_ceara.png")
            print("Imagem salva como queimadas_ai_ceara.png")
            return f"Sucesso AI: {count} anomalias detectadas. Imagem recortada salva."
        else:
            return "Erro: Máscara do Ceará vazia ao tentar recortar."
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Erro AI: {str(e)}"

# 4. EXECUÇÃO DO AGENTE (PADRÃO TOOL CALLING)
def run():
    print(f"\n🤖 Iniciando Agente...")
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    tools = [download_goes_data, analyze_fire]
    
    # Prompt Moderno (Obrigatório para tool_calling)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Você é um especialista em satélites."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    # Criação do Agente (Esta função substitui as antigas que deram erro)
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Solicita dados de uma data conhecida (Ex: 05/04/2025 15h) pois 2026 não tem dados ainda
    agent_executor.invoke({"input": "Verifique queimadas no GOES-16 para a data 2025-04-05 15:00 UTC."})

if __name__ == "__main__":
    run()
