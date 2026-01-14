# Monitor de Queimadas GOES-16 (Ceará) - AI & LangGraph 🛰️🔥

Este projeto é um sistema avançado de monitoramento de incêndios florestais focado no estado do Ceará, Brasil. Ele utiliza dados brutos do satélite GOES-16 (NOAA), processamento de imagens com Inteligência Artificial Não-Supervisionada (K-Means) e um agente especialista baseado em LangGraph para orquestração e análise.

## 🚀 Funcionalidades

1.  **Coleta de Dados de Satélite**: Download automático de bandas espectrais (07 - Infravermelho Curto e 13 - Infravermelho Limpo) do bucket AWS S3 do NOAA (`noaa-goes16`).
2.  **Detecção de Fogo com IA (Unsupervised)**:
    *   Algoritmo **K-Means Clustering** (`scikit-learn`) para segmentar a imagem termal em clusters (Nuvem, Terra, Fogo).
    *   Refinamento estatístico para reduzir falsos positivos (Filtro de Temperatura > 315K0).
3.  **Mapeamento Oficial**: Integração com a API do IBGE (`geopandas`) para plotar os focos confirmados sobre o mapa oficial do Ceará.
4.  **Agente Especialista (LangGraph)**:
    *   Fluxo de trabalho orquestrado que executa o pipeline de dados -> detecção -> mapeamento.
    *   Utiliza **GPT-4o** para gerar um *Parecer Técnico* automático analisando a severidade das queimadas.
5.  **Dashboard Interativo**: Interface web construída com **Streamlit** para visualização fácil e seleção de datas históricas.

## 🛠️ Instalação

### Pré-requisitos
- Python 3.10+
- Chave da OpenAI (`OPENAI_API_KEY`) no arquivo `.env`.

### Passos

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/seu-usuario/monitor-queimadas-ceara.git
    cd monitor-queimadas-ceara
    ```

2.  **Crie e ative um ambiente virtual:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    # venv\Scripts\activate   # Windows
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure as variáveis de ambiente:**
    Crie um arquivo `.env` na raiz:
    ```env
    OPENAI_API_KEY=sk-sua-chave-aqui
    ```

## 🖥️ Como Usar

### Executar o Dashboard

O comando principal para iniciar a aplicação é:

```bash
streamlit run dashboard.py
```

Acesse **http://localhost:8501** no seu navegador.

1.  Selecione a **Data** e **Hora** no menu lateral.
2.  Clique em **"Iniciar Análise Especialista"**.
3.  O sistema irá:
    *   Baixar os dados históricos do GOES-16.
    *   Rodar a IA para detectar anomalias.
    *   Buscar o mapa atualizado do IBGE.
    *   Gerar o parecer técnico.

### Estrutura do Projeto

*   `dashboard.py`: Interface do usuário (Frontend).
*   `agent_graph.py`: Lógica do Agente LangGraph (Backend + IA).
*   `queimadas_goes16.py`: Ferramentas de baixo nível (Download S3, K-Means).
*   `data/`: Diretório temporário para arquivos NetCDF (ignorado no git).

## 📊 Tecnologias

*   **Python**: Linguagem principal.
*   **LangGraph & LangChain**: Orquestração de agentes.
*   **Scikit-Learn**: Machine Learning (K-Means).
*   **Geopandas**: Manipulação de dados geoespaciais e Shapefiles.
*   **Streamlit**: Visualização de dados.
*   **AWS Boto3**: Acesso aos dados do satélite.

---
Desenvolvido como prova de conceito para monitoramento ambiental inovador.
