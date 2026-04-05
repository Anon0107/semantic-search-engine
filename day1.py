import voyageai
from dotenv import load_dotenv
import math
import numpy as np

load_dotenv()
vo = voyageai.Client()

sentences = [
    "Bocchi the Rock is about a shy guitarist",
    "Claude is an AI assistant made by Anthropic",
    "BanG Dream! It's MyGO!!!!! was released in 2023",
    "BanG Dream! Ave Mujica was released in 2025",
    "Anon Chihaya and Rana Kaname are the guitarists of MyGO!!!!!",
    "Sakiko Togawa is the initiater of Ave Mujica",
    "Netherite is the strongest material for equipments in Minecraft",
    "Opus 4.6 is Claude's best model",
    "Kasumi Toyama is the vocal of Poppin'Party",
    "Sakiko Togawa and Uika Misumi is related in blood",
    "Diamond blocks can be naturally generated in Minecraft",
    "Mashiro Kurata is the vocal of Morfonica",
    "Mahiru Shiina is the female main character of Otonari no Tenshisama",
    "Amane Fujimiya is the male main character of Otonari no Tenshisama",
    "Anon Chihaya is pink haired",
    "Rana Kaname is white haired",
    "Soyo Nagasaki is plays base for MyGO!!!!!",
    "Soyo Nagasaki, Mutsumi Wakaba and Mashiro Kurata are from the same school",
    "Mutsumi Wakaba and Uika Misimu are the guitarists of Ave Mujica",
    "Haiku is Claude's fastest and cheapest model"
]

result = vo.embed(sentences, model="voyage-3", input_type="document")
vectors = result.embeddings

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot_product = 0
    sum_square_a = 0
    sum_square_b = 0
    for num_a, num_b in zip(a,b):
        sum_square_a += num_a**2
        sum_square_b += num_b**2
        dot_product += num_a*num_b
    magnitude_a = math.sqrt(sum_square_a)
    magnitude_b = math.sqrt(sum_square_b)
    return dot_product / (magnitude_a*magnitude_b)

def cosine_similarity_np(a: list[float], b: list[float]) -> float:
    a = np.array(a)
    b = np.array(b)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

print('Cosine similarities using brute force math:')
print('Mahiru & Amane, same show — highest, same anime same sentence structure')
print(cosine_similarity(vectors[12],vectors[13]))
print('Kasumi Toyama & Mashiro Kurata, both BanG Dream vocalists — high, related topic')
print(cosine_similarity(vectors[8],vectors[11]))
print('MyGO 2023 release & Anon/Rana are guitarists — moderate, same band different facts')
print(cosine_similarity(vectors[2],vectors[4]))
print('Bocchi the Rock & Haiku is cheapest model — lowest, unrelated topics')
print(cosine_similarity(vectors[0],vectors[19]))
print('Cosine similarities using numpy:')
print(cosine_similarity_np(vectors[2],vectors[4]))
print(cosine_similarity_np(vectors[12],vectors[13]))
print(cosine_similarity_np(vectors[0],vectors[19]))
print(cosine_similarity_np(vectors[8],vectors[11]))
