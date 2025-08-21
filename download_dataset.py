from datasets import load_dataset

# Caminho onde o dataset será armazenado
cache_dir = r"C:\Users\Admin\Documents\Site MostraVIP\API-L-menAI\API-L-menAI"

# Carregando dataset e forçando o download para o cache_dir especificado
ds = load_dataset("cnmoro/reasoning-v1-20m-portuguese", cache_dir=cache_dir)

print(ds)
