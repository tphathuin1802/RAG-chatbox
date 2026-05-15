import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# If you later switch demo code to LangChain/OpenAI, ensure .env works
# even when running from a different directory.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
    if not os.getenv("OPENAI_API_KEY") and os.getenv("OPEN_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["OPEN_API_KEY"]
except Exception:
    # demo.py can run without dotenv installed / without OpenAI usage
    pass

# Tải mô hình embedding (mô hình này tạo ra vector 384 chiều)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Tập dữ liệu mẫu
sentences = [
    "Tôi đang xây dựng một 파이프라인 Machine Learning.", # Câu 0
    "Trí tuệ nhân tạo và hệ thống dữ liệu rất thú vị.",    # Câu 1
    "Hôm nay thời tiết bên ngoài có vẻ sắp mưa."         # Câu 2
]

# Chuyển đổi văn bản thành không gian vector
embeddings = model.encode(sentences)

# Kiểm tra số chiều của không gian vector
print(f"Kích thước của mỗi vector: {embeddings.shape[1]} chiều\n")

# Tính toán khoảng cách (Cosine Similarity)
sim_0_1 = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
sim_0_2 = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]

print(f"Độ tương đồng giữa Câu 0 và Câu 1 (cùng lĩnh vực kỹ thuật): {sim_0_1:.4f}")
print(f"Độ tương đồng giữa Câu 0 và Câu 2 (khác lĩnh vực): {sim_0_2:.4f}")