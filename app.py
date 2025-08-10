import os
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
import json
import difflib
import unidecode
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import threading
import time
import requests

app = Flask(__name__)
CORS(app)  # habilita CORS para todas as rotas

# --- Arquivos JSON base com perguntas e respostas fixas ---
arquivos_json = [
    "perguntas_respostas.json",
    "perguntas_respostas2.json",
    "perguntas_respostas3.json",
    "perguntas_respostas4.json",
    "perguntas_respostas5.json",
    "perguntas_respostas6.json"
]

base_qa = []

# Carrega as bases de perguntas e respostas
for arquivo in arquivos_json:
    try:
        with open(arquivo, "r", encoding="utf-8") as f:
            base_qa.extend(json.load(f))
    except FileNotFoundError:
        print(f"Aviso: arquivo {arquivo} não encontrado. Ignorando.")
    except json.JSONDecodeError:
        print(f"Erro ao ler JSON no arquivo {arquivo}. Verifique o formato.")

def salvar_base_em_arquivo(arquivo):
    try:
        with open(arquivo, "w", encoding="utf-8") as f:
            json.dump(base_qa, f, ensure_ascii=False, indent=2)
        print(f"Base salva em {arquivo}.")
    except Exception as e:
        print(f"Erro ao salvar base em {arquivo}: {e}")

# --- Função para limpar texto (remover acentos, pontuação e deixar minúsculo) ---
def limpar_texto(texto):
    texto = texto.lower()
    texto = unidecode.unidecode(texto)
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    return texto

# --- Busca por similaridade com difflib na base QA ---
def buscar_por_similaridade(pergunta):
    p_limpa = limpar_texto(pergunta)
    perguntas_limpa = [limpar_texto(item["pergunta"]) for item in base_qa]
    melhor_match = difflib.get_close_matches(p_limpa, perguntas_limpa, n=1, cutoff=0.85)
    if melhor_match:
        index = perguntas_limpa.index(melhor_match[0])
        return base_qa[index]["resposta"]
    return None

# --- Carregamento das intents ---
try:
    with open("intents.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        intents = data.get("intents", [])
except FileNotFoundError:
    print("Arquivo intents.json não encontrado. Intents desabilitados.")
    intents = []

# Monta lista de exemplos (keywords) e mapeia para intents para TF-IDF
exemplos = []
mapping_intent = []
for intent in intents:
    if isinstance(intent, dict):
        for kw in intent.get("keywords", []):
            exemplos.append(limpar_texto(kw))
            mapping_intent.append(intent.get("intent", "unknown"))
    else:
        print(f"Warning: ignorando intent inválido (não é dict): {intent}")

# Inicializa TF-IDF
if exemplos:
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(exemplos)
else:
    vectorizer = None
    X = None

# --- Busca por intents ---
def buscar_por_intent(pergunta):
    if not vectorizer or X is None:
        return None, None

    p_limpa = limpar_texto(pergunta)
    palavras_pergunta = set(p_limpa.split())

    # 1) Busca pela intent com maior número de palavras-chave presentes na pergunta
    max_matches = 0
    melhor_intent = None

    for intent in intents:
        keywords = [limpar_texto(kw) for kw in intent.get("keywords", [])]
        matches = sum(1 for kw in keywords if kw in palavras_pergunta)
        if matches > max_matches:
            max_matches = matches
            melhor_intent = intent

    if melhor_intent and max_matches > 0:
        return melhor_intent.get("response", ""), None

    # 2) Busca por TF-IDF + similaridade cosseno, como fallback
    v_pergunta = vectorizer.transform([p_limpa])
    similares = cosine_similarity(v_pergunta, X)
    idx = similares.argmax()
    similaridade = similares[0, idx]

    if similaridade > 0.5:
        intent_nome = mapping_intent[idx]
        for intent in intents:
            if intent.get("intent") == intent_nome:
                return intent.get("response", ""), None

    return None, None

# --- Endpoint principal para perguntas ---
@app.route("/pergunta", methods=["POST"])
def responder():
    data = request.get_json()
    pergunta = data.get("pergunta", "").strip()

    if not pergunta:
        return jsonify({"erro": "Informe a pergunta no formato JSON {\"pergunta\": \"sua pergunta aqui\"}"}), 400

    # 1) Tenta buscar na base QA com difflib
    resposta = buscar_por_similaridade(pergunta)
    if resposta:
        return jsonify({"resposta": resposta})

    # 2) Tenta buscar nas intents (palavra-chave ou TF-IDF)
    resposta, template = buscar_por_intent(pergunta)
    if resposta:
        # Se template existe, substitui {resposta} pelo texto da resposta
        if template and "{resposta}" in template:
            resposta = template.replace("{resposta}", resposta)
        # Caso não tenha template ou não tenha a variável, envia só a resposta
        return jsonify({"resposta": resposta})

    # 3) Fallback padrão
    return jsonify({"resposta": "Desculpe, não entendi. Pode perguntar de outro jeito?"})

# --- Endpoint para adicionar pergunta/resposta nova ---
@app.route("/adicionar", methods=["POST"])
def adicionar_pergunta_resposta():
    data = request.get_json()
    pergunta = data.get("pergunta", "").strip()
    resposta = data.get("resposta", "").strip()

    if not pergunta or not resposta:
        return jsonify({"erro": "Envie JSON com 'pergunta' e 'resposta' preenchidos."}), 400

    # Evita duplicatas
    for item in base_qa:
        if item["pergunta"].lower() == pergunta.lower():
            return jsonify({"erro": "Essa pergunta já existe na base."}), 400

    base_qa.append({"pergunta": pergunta, "resposta": resposta})
    salvar_base_em_arquivo(arquivos_json[0])  # salva na primeira base

    return jsonify({"msg": "Pergunta e resposta adicionadas com sucesso!"})

# --- Endpoint para manter o servidor “acordado” protegido por token ---
PING_TOKEN = os.getenv("PING_TOKEN")

@app.route("/ping", methods=["GET"])
def ping():
    token = request.headers.get("Authorization")
    if PING_TOKEN and token != f"Bearer {PING_TOKEN}":
        abort(401)
    return jsonify({"status": "alive"})

# --- Thread para autoping usando URL pública e token ---
def autoping():
    url = os.getenv("PING_URL")  # Exemplo: https://meuapp.onrender.com/ping
    token = os.getenv("PING_TOKEN")
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    while True:
        try:
            response = requests.get(url, headers=headers)
            print(f"Autoping: {response.status_code} - {response.json()}")
        except Exception as e:
            print(f"Erro no autoping: {e}")
        time.sleep(300)  # espera 5 minutos

if __name__ == "__main__":
    threading.Thread(target=autoping, daemon=True).start()
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
