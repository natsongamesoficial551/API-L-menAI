import os
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
import json
import difflib
import unidecode
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import threading
import time
import requests
import glob

app = Flask(__name__)
CORS(app)

# --- Arquivos JSON base ---
arquivos_json = [
    "perguntas_respostas.json",
    "perguntas_respostas2.json",
    "perguntas_respostas3.json",
    "perguntas_respostas4.json",
    "perguntas_respostas5.json",
    "perguntas_respostas6.json"
]

base_qa = []

for arquivo in arquivos_json:
    try:
        with open(arquivo, "r", encoding="utf-8") as f:
            base_qa.extend(json.load(f))
    except Exception as e:
        print(f"Aviso ao carregar {arquivo}: {e}")

def salvar_base_em_arquivo(arquivo):
    try:
        with open(arquivo, "w", encoding="utf-8") as f:
            json.dump(base_qa, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Erro ao salvar base: {e}")

# --- Função utilitária ---
def limpar_texto(texto):
    texto = texto.lower()
    texto = unidecode.unidecode(texto)
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    return texto

# --- Busca difflib ---
def buscar_por_similaridade(pergunta):
    p_limpa = limpar_texto(pergunta)
    perguntas_limpa = [limpar_texto(item["pergunta"]) for item in base_qa]
    match = difflib.get_close_matches(p_limpa, perguntas_limpa, n=1, cutoff=0.70)
    if match:
        idx = perguntas_limpa.index(match[0])
        return base_qa[idx]["resposta"]
    return None

# --- Carregamento intents ---
arquivos_intents = glob.glob("intents*.json")  # pega todos os intents*.json
intents = []

for arquivo in arquivos_intents:
    try:
        with open(arquivo, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "intents" in data:
                intents.extend(data["intents"])
            elif isinstance(data, list):
                intents.extend(data)
    except Exception as e:
        print(f"Aviso ao carregar intent {arquivo}: {e}")

# --- TF-IDF ---
exemplos = []
mapping_intent = []
for intent in intents:
    for kw in intent.get("patterns", []):
        exemplos.append(limpar_texto(kw))
        mapping_intent.append(intent.get("intent", "unknown"))

vectorizer = TfidfVectorizer() if exemplos else None
X = vectorizer.fit_transform(exemplos) if exemplos else None

# --- Busca por intents com TF-IDF + cosine similarity ---
def buscar_por_intent(pergunta):
    if not vectorizer or X is None:
        return None

    p_limpa = limpar_texto(pergunta)

    # TF-IDF + cosseno
    v_pergunta = vectorizer.transform([p_limpa])
    similares = cosine_similarity(v_pergunta, X)
    idx = similares.argmax()
    similaridade = similares[0, idx]

    # Se a similaridade for boa, retorna a resposta
    if similaridade >= 0.6:  # chute mais flexível
        intent_nome = mapping_intent[idx]
        for intent in intents:
            if intent.get("intent") == intent_nome:
                respostas = intent.get("responses", [])
                if respostas:
                    return respostas[0]
    return None

# --- Endpoint principal ---
@app.route("/pergunta", methods=["POST"])
def responder():
    data = request.get_json()
    pergunta = data.get("pergunta", "").strip()
    if not pergunta:
        return jsonify({"erro": "Informe a pergunta no formato JSON {\"pergunta\": \"sua pergunta aqui\"}"}), 400

    # 1) Difflib QA
    resposta = buscar_por_similaridade(pergunta)
    if resposta:
        return jsonify({"resposta": resposta})

    # 2) Intents
    resposta = buscar_por_intent(pergunta)
    if resposta:
        return jsonify({"resposta": resposta})

    return jsonify({"resposta": "Desculpe, não entendi. Pode perguntar de outro jeito?"})

# --- Endpoint adicionar ---
@app.route("/adicionar", methods=["POST"])
def adicionar_pergunta_resposta():
    data = request.get_json()
    pergunta = data.get("pergunta", "").strip()
    resposta = data.get("resposta", "").strip()
    if not pergunta or not resposta:
        return jsonify({"erro": "Envie JSON com 'pergunta' e 'resposta' preenchidos."}), 400

    if any(item["pergunta"].lower() == pergunta.lower() for item in base_qa):
        return jsonify({"erro": "Pergunta já existe."}), 400

    base_qa.append({"pergunta": pergunta, "resposta": resposta})
    salvar_base_em_arquivo(arquivos_json[0])
    return jsonify({"msg": "Pergunta e resposta adicionadas!"})

# --- Ping ---
PING_TOKEN = os.getenv("PING_TOKEN")
@app.route("/ping", methods=["GET"])
def ping():
    token = request.headers.get("Authorization")
    if PING_TOKEN and token != f"Bearer {PING_TOKEN}":
        abort(401)
    return jsonify({"status": "alive"})

# --- Autoping ---
def autoping():
    url = os.getenv("PING_URL")
    token = os.getenv("PING_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    while True:
        try:
            response = requests.get(url, headers=headers)
            print(f"Autoping: {response.status_code}")
        except Exception as e:
            print(f"Erro autoping: {e}")
        time.sleep(300)

if __name__ == "__main__":
    threading.Thread(target=autoping, daemon=True).start()
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
