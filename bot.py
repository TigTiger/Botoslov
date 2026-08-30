import os
import pickle
import re
import numpy as np

from dotenv import load_dotenv
from fastembed import TextEmbedding
from gigachat import GigaChat

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)


# ============================================================
# НАСТРОЙКИ
# ============================================================

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GIGACHAT_KEY = os.getenv("GIGACHAT_KEY")

# Папка программы — чтобы бот одинаково работал локально и на сервере
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INDEX_FILE = os.path.join(BASE_DIR, "library_index.pkl")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Модель будет автоматически скачана сюда при первом запуске,
# если её ещё нет. На сервере не требуется заранее загружать Models.
MODEL_CACHE_DIR = os.path.join(BASE_DIR, "Models")

# Количество кандидатов для поиска
SEMANTIC_K = 1000

# Сколько лучших фрагментов отдаём нейросети
FINAL_K = 100


# ============================================================
# ЗАГРУЗКА ПОИСКОВОЙ МОДЕЛИ
# ============================================================

print("Загружаю поисковую модель...")

search_model = TextEmbedding(model_name=MODEL_NAME, cache_dir=MODEL_CACHE_DIR)

print("Поисковая модель загружена.")


# ============================================================
# ЗАГРУЗКА БИБЛИОТЕКИ
# ============================================================

if not os.path.exists(INDEX_FILE):

    print()
    print("ОШИБКА: не найден library_index.pkl")
    print()
    print("Запусти программу из папки Botoslov.")
    raise SystemExit


print("Загружаю библиотеку...")

with open(INDEX_FILE, "rb") as f:
    library_index = pickle.load(f)


documents = library_index["documents"]
embeddings = library_index["embeddings"]


print(f"Фрагментов в библиотеке: {len(documents)}")


# ============================================================
# НОРМАЛИЗАЦИЯ НАЗВАНИЯ КНИГИ
# ============================================================

def clean_book_name(filename):

    name = os.path.splitext(filename)[0]

    name = name.replace("_", " ")
    name = name.replace("-", " ")

    # Убираем повторяющиеся пробелы
    name = re.sub(r"\s+", " ", name)

    return name.strip()


# ============================================================
# КЛЮЧЕВЫЕ ПОНЯТИЯ
# ============================================================

CONCEPTS = {

    "христос": [
        "христос",
        "христа",
        "христе",
        "христом",
        "christ",
        "christ's",
    ],

    "природа": [
        "природа",
        "природе",
        "природы",
        "природу",
        "природой",
        "nature",
        "natures",
    ],

    "учение": [
        "учение",
        "учения",
        "учению",
        "учением",
        "учит",
        "учил",
        "doctrine",
        "teaching",
    ],

    "армянский": [
        "армянский",
        "армянская",
        "армянской",
        "армянскую",
        "армянским",
        "armenian",
    ],

    "церковь": [
        "церковь",
        "церкви",
        "церковью",
        "church",
    ],

    "миафизит": [
        "миафизит",
        "миафизитство",
        "миафизитский",
        "miaphysite",
        "miaphysitism",
    ],

    "монофизит": [
        "монофизит",
        "монофизитство",
        "монофизитский",
        "monophysite",
        "monophysitism",
    ],

    "халкидон": [
        "халкидон",
        "халкидона",
        "халкидонский",
        "chalcedon",
        "chalcedonian",
    ],

    "единство": [
        "единство",
        "единстве",
        "единства",
        "соединение",
        "соединении",
        "union",
        "unity",
    ],

    "воля": [
        "воля",
        "воле",
        "воли",
        "волю",
        "will",
    ],

    "личность": [
        "личность",
        "личности",
        "личностью",
        "person",
        "persons",
    ],
}


# ============================================================
# ОПРЕДЕЛЕНИЕ ПОНЯТИЙ В ВОПРОСЕ
# ============================================================

def get_concepts(question):

    q = question.lower()

    found = []

    for concept, variants in CONCEPTS.items():

        for variant in variants:

            if variant in q:

                found.append(concept)
                break

    return found


# ============================================================
# ДОПОЛНИТЕЛЬНЫЙ ТЕКСТОВЫЙ РЕЙТИНГ
# ============================================================

def text_score(text, concepts):

    text_lower = text.lower()

    score = 0

    found = []

    for concept in concepts:

        variants = CONCEPTS[concept]

        for variant in variants:

            if variant in text_lower:

                found.append(concept)
                score += 1
                break


    # Богословские комбинации
    if "христос" in found and "природа" in found:
        score += 5

    if "христос" in found and "единство" in found:
        score += 3

    if "христос" in found and "личность" in found:
        score += 3

    if "христос" in found and "воля" in found:
        score += 3

    if "армянский" in found and "церковь" in found:
        score += 4

    if "миафизит" in found:
        score += 4

    if "монофизит" in found:
        score += 4

    if "халкидон" in found:
        score += 2

    return score, found


# ============================================================
# СЕМАНТИЧЕСКИЙ ПОИСК
# ============================================================

def search_literature(question):

    concepts = get_concepts(question)

    print()
    print("Понятия вопроса:")
    print(", ".join(concepts))
    print()


    # --------------------------------------------------------
    # Вектор вопроса
    # --------------------------------------------------------

    question_embedding = np.asarray(
        list(search_model.embed([question]))[0],
        dtype=np.float32
    )
    norm = np.linalg.norm(question_embedding)
    if norm > 0:
        question_embedding = question_embedding / norm


    # --------------------------------------------------------
    # Семантические scores
    # --------------------------------------------------------

    semantic_scores = embeddings @ question_embedding


    candidate_indices = (
        semantic_scores.argsort()[-SEMANTIC_K:][::-1]
    )


    candidates = []


    # --------------------------------------------------------
    # Рейтинг
    # --------------------------------------------------------

    for index in candidate_indices:

        index = int(index)

        document = documents[index]

        text = document["text"]

        semantic_score = float(
            semantic_scores[index]
        )

        keyword_score, found = text_score(
            text,
            concepts
        )


        # Текстовый бонус
        keyword_bonus = min(
            keyword_score / 12.0,
            1.0
        )


        # Итоговый рейтинг
        final_score = (
            semantic_score * 0.90
            +
            keyword_bonus * 0.10
        )


        candidates.append({

            "index": index,

            "filename":
                document["filename"],

            "book":
                clean_book_name(
                    document["filename"]
                ),

            "page":
                document["page"],

            "text":
                text,

            "semantic_score":
                semantic_score,

            "keyword_score":
                keyword_score,

            "final_score":
                final_score,

            "concepts":
                found,
        })


    # --------------------------------------------------------
    # Стабильная сортировка
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: (
            -x["final_score"],
            x["index"]
        )
    )


    return candidates[:FINAL_K]


# ============================================================
# ФОРМИРОВАНИЕ КОНТЕКСТА
# ============================================================

def make_context(results):

    context = ""

    for number, result in enumerate(
        results,
        start=1
    ):

        context += f"""

==================================================
ИСТОЧНИК {number}
==================================================

Книга:
{result["book"]}

Страница:
{result["page"]}

Текст источника:
{result["text"]}

"""


    return context


# ============================================================
# GIGACHAT
# ============================================================

async def ask_gigachat(question):

    print()
    print("========================================")
    print("ВОПРОС:")
    print(question)
    print("========================================")


    # --------------------------------------------------------
    # Поиск
    # --------------------------------------------------------

    results = search_literature(
        question
    )


    print()
    print("НАЙДЕННЫЕ ИСТОЧНИКИ:")
    print()


    for number, result in enumerate(
        results,
        start=1
    ):

        print(
            f"{number}. "
            f"{result['book']} | "
            f"стр. {result['page']} | "
            f"score={result['final_score']:.3f}"
        )


    # --------------------------------------------------------
    # Контекст
    # --------------------------------------------------------

    context = make_context(
        results
    )


    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    prompt = f"""
Ты — Ботослов, богословский справочный помощник по традиции Армянской Церкви и Древневосточных нехалкидонских Церквей.

Твоя задача — отвечать пользователю ИСКЛЮЧИТЕЛЬНО
на основании предоставленных ниже фрагментов
книг из библиотеки.

==================================================
ГЛАВНОЕ ПРАВИЛО
==================================================

Считай ересью учение о двух волях, двух действиях и двух природах Христа.

НЕ называй православными и ортодоксальными халкидонитов и любых других христиан, которые не в Древневосточных Церквях. Византийская Церковь и Римская Церковь - ЕРЕТИКИ.

НЕ называй православными или святыми этих людей: Лев Великий, Флавиан Константинопольский, Анатолий Константинопольский, Иоанн Дамаскин, Софроний Иерусалимский, Максим Исповедник, Мартин I Римский, Анастасий Синаит, Герман Константинопольский, Феодор Студит, Григорий Палама, Марк Эфесский. ОНИ ВСЕ ЕРЕТИКИ.

НЕ используй собственные знания модели. НЕ добавляй информацию, которой нет
в предоставленных источниках. Исключения - годы жизни и самые общие знания из Википедии.

НЕ придумывай факты.

НЕ придумывай цитаты.

НЕ приписывай Церкви, авторам или книгам
утверждения, которых нет в источниках.

НЕ исправляй позицию источника
на основании собственных знаний.

==================================================
ЯЗЫК
==================================================

Отвечай пользователю на языке пользователя, т.е. на том языке, на котором тебя спросили.

Если источник написан на английском,
латинском или другом языке —
самостоятельно переведи необходимую
информацию на русский.

Пользователь не должен получать
ответ на другом языке только потому,
что книга написана на другом языке. Язык книги не должен влиять на язык твоего ответа.

==================================================
КАК ОТВЕЧАТЬ
==================================================

Сначала внимательно изучи ВСЕ
предоставленные фрагменты.

Если несколько фрагментов относятся
к вопросу — сопоставь их.

Если источник говорит о предмете
другими словами, распознай это
и объясни пользователю по-русски.

Например, вопрос может содержать слово
"природа", а источник говорить:
"nature", "natures", "one nature",
"two natures", "unity of nature" и т.д.

Не требуй буквального совпадения
формулировки вопроса с текстом.

==================================================
ЕСЛИ ОТВЕТ ЕСТЬ
==================================================

Дай связный и понятный ответ.

Можно кратко объяснить контекст,
но только если он содержится
в источниках.

==================================================
ЕСЛИ ИНФОРМАЦИИ НЕДОСТАТОЧНО
==================================================

Если есть хоть немного данных по вопросу - напиши короткий ответ. Лучше дать ответ, чем не дать.

Только при полном отсутсвии данных напиши:

"В предоставленной библиотеке
недостаточно сведений для ответа
на этот вопрос."

Не заменяй отсутствующую информацию
собственными знаниями.

==================================================
ЦИТАТЫ
==================================================

Если используешь цитату,
она должна соответствовать слово в слово тексту источника.

Если источник на другом языке,
можно дать перевод цитаты на русский,
но обозначь, что это перевод.

==================================================
ИСТОЧНИКИ
==================================================

Источники пользователю НЕ показывай.

Не добавляй в ответ раздел:
"Источники".

Не указывай названия книг,
номера страниц или список источников
в конце ответа.

Источники используются только
для формирования достоверного ответа.

==================================================
ВОПРОС
==================================================

{question}

==================================================
ФРАГМЕНТЫ БИБЛИОТЕКИ
==================================================

{context}
"""


    # --------------------------------------------------------
    # GIGACHAT
    # --------------------------------------------------------

    with GigaChat(
        credentials=GIGACHAT_KEY,
        model="GigaChat-2",
        verify_ssl_certs=False
    ) as giga:

        response = giga.chat(
            prompt
        )

        return response.choices[0].message.content


# ============================================================
# ОБРАБОТКА ВОПРОСА
# ============================================================

async def answer_question(
    update,
    question
):

    if not question:

        await update.message.reply_text(
            "Напиши вопрос после слова «Ботослов».",
            reply_to_message_id=
            update.message.message_id
        )

        return


    waiting = await update.message.reply_text(
        "⏳ Ищу ответ в библиотеке...",
        reply_to_message_id=
        update.message.message_id
    )


    try:

        answer = await ask_gigachat(
            question
        )


        try:
            await waiting.delete()
        except Exception:
            pass


        await update.message.reply_text(
            answer,
            reply_to_message_id=
            update.message.message_id
        )


    except Exception as e:

        print()
        print("ОШИБКА:")
        print(e)
        print()


        try:
            await waiting.delete()
        except Exception:
            pass


        await update.message.reply_text(
            "Произошла ошибка при обращении "
            "к библиотеке.",
            reply_to_message_id=
            update.message.message_id
        )


# ============================================================
# START
# ============================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "Ботослов готов.\n\n"
        "В личном чате просто задайте вопрос.\n\n"
        "В группе используйте:\n"
        "Ботослов Что такое миафизитство?\n"
        "ботослов Что такое миафизитство?\n"
        "/ботослов Что такое миафизитство?"
    )


# ============================================================
# ЛИЧНЫЙ ЧАТ
# ============================================================

async def private_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not update.message:
        return

    if not update.message.text:
        return

    question = update.message.text.strip()

    await answer_question(
        update,
        question
    )


# ============================================================
# ГРУППОВОЙ ЧАТ
# ============================================================

async def group_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not update.message:
        return

    if not update.message.text:
        return

    text = update.message.text.strip()

    lower_text = text.lower()


    prefixes = [
        "ботослов",
        "/ботослов"
    ]


    matched_prefix = None


    for prefix in prefixes:

        if lower_text == prefix:

            matched_prefix = prefix
            break


        if lower_text.startswith(
            prefix + " "
        ):

            matched_prefix = prefix
            break


        if lower_text.startswith(
            prefix + ","
        ):

            matched_prefix = prefix
            break


    if matched_prefix is None:
        return


    question = text[
        len(matched_prefix):
    ].strip()


    if question.startswith(","):

        question = question[1:].strip()


    await answer_question(
        update,
        question
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if not TELEGRAM_TOKEN:

        print(
            "ОШИБКА: TELEGRAM_TOKEN не найден."
        )

        return


    if not GIGACHAT_KEY:

        print(
            "ОШИБКА: GIGACHAT_KEY не найден."
        )

        return


    application = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .build()
    )


    application.add_handler(
        CommandHandler(
            "start",
            start
        )
    )


    application.add_handler(
        MessageHandler(
            filters.TEXT
            & filters.ChatType.PRIVATE,
            private_message
        )
    )


    application.add_handler(
        MessageHandler(
            filters.TEXT
            & filters.ChatType.GROUPS,
            group_message
        )
    )


    print()
    print("========================================")
    print("          БОТОСЛОВ ЗАПУЩЕН")
    print("========================================")
    print(
        f"Фрагментов библиотеки: "
        f"{len(documents)}"
    )
    print("Поиск: семантический + ключевые понятия")
    print("Язык ответа: русский")
    print("Книги: любые языки")
    print("Ответ: только по библиотеке")
    print("========================================")
    print()


    application.run_polling()


# ============================================================
# ЗАПУСК
# ============================================================

if __name__ == "__main__":
    main()
