import os
import pickle
import sys
import numpy as np
from fastembed import TextEmbedding

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIBRARY_FILE = os.path.join(BASE_DIR, "library_text.pkl")
INDEX_FILE = os.path.join(BASE_DIR, "library_index.pkl")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_CACHE_DIR = os.path.join(BASE_DIR, "Models")
BATCH_SIZE = 32


def load_documents():
    if not os.path.exists(LIBRARY_FILE):
        print("ОШИБКА: не найден library_text.pkl")
        sys.exit(1)

    with open(LIBRARY_FILE, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        if "documents" in data:
            documents = data["documents"]
        elif "library" in data:
            documents = data["library"]
        else:
            documents = []
            for filename, value in data.items():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            item = dict(item)
                            item.setdefault("filename", filename)
                            documents.append(item)
    elif isinstance(data, list):
        documents = data
    else:
        raise ValueError("Неизвестный формат library_text.pkl")

    cleaned = []
    for d in documents:
        if not isinstance(d, dict) or "text" not in d:
            continue
        item = dict(d)
        item.setdefault("filename", item.get("file", "Неизвестная книга"))
        item.setdefault("page", item.get("page_number", "?"))
        item["text"] = str(item["text"]).strip()
        if item["text"]:
            cleaned.append(item)

    if not cleaned:
        raise ValueError("В library_text.pkl нет фрагментов с текстом")

    return cleaned


def main():
    print("=" * 40)
    print("СОЗДАНИЕ НОВОГО ИНДЕКСА")
    print("=" * 40)

    documents = load_documents()
    print(f"Фрагментов: {len(documents)}")
    print(f"Модель: {MODEL_NAME}")
    print()

    model = TextEmbedding(
        model_name=MODEL_NAME,
        cache_dir=MODEL_CACHE_DIR
    )

    texts = [d["text"] for d in documents]
    all_embeddings = []

    print("Создаю embeddings...")

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]

        for emb in model.embed(batch):
            arr = np.asarray(emb, dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            all_embeddings.append(arr)

        done = min(start + BATCH_SIZE, len(texts))
        print(f"Обработано: {done}/{len(texts)}")

    embeddings = np.asarray(all_embeddings, dtype=np.float32)

    if len(embeddings) != len(documents):
        raise RuntimeError("Количество embeddings не совпадает с количеством фрагментов")

    result = {
        "documents": documents,
        "embeddings": embeddings,
        "embedding_model": MODEL_NAME,
        "embedding_backend": "fastembed/onnxruntime",
    }

    with open(INDEX_FILE, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    print()
    print("=" * 40)
    print("ИНДЕКС БИБЛИОТЕКИ СОЗДАН!")
    print("=" * 40)
    print(f"Фрагментов: {len(documents)}")
    print(f"Размерность: {embeddings.shape[1]}")
    print(f"Файл: {INDEX_FILE}")
    print("=" * 40)


if __name__ == "__main__":
    main()
