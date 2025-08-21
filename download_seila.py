from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Nome do modelo
model_name = "CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it"

# Carrega o tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Carrega o modelo em 8-bit com offload para CPU quando necessário
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,                     # quantização 8-bit
    device_map="auto",                     # gerencia automaticamente GPU/CPU
    llm_int8_enable_fp32_cpu_offload=True, # offload seguro para CPU
    trust_remote_code=True
)

# Pipeline de geração de texto
# OBS: NÃO passar device aqui quando usamos device_map="auto"
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# Prompt de teste
prompt = "Explique de forma simples o que é uma metáfora."

# Geração de texto
result = pipe(prompt, max_new_tokens=100)

# Mostra o resultado
print(result[0]['generated_text'])
