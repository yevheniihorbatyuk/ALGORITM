from text_analysis_system import Document
from text_analysis_advanced import create_demo_system
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_advanced_demo():
    """Демонстрація розширених можливостей системи"""
    logger.info("Ініціалізація системи...")
    search_engine, semantic_search, search_optimizer = create_demo_system()
    
    # Створюємо тестові документи
    documents = [
        Document(
            id="1",
            title="Машинне навчання: основи",
            content="Машинне навчання - це підгалузь штучного інтелекту, "
                   "яка фокусується на розробці алгоритмів, що можуть навчатися з даних.",
            keywords={"ML", "AI", "алгоритми"}
        ),
        Document(
            id="2",
            title="Нейронні мережі в комп'ютерному зорі",
            content="Згорткові нейронні мережі (CNN) широко використовуються "
                   "для розпізнавання образів та комп'ютерного зору.",
            keywords={"CNN", "AI", "комп'ютерний зір"}
        ),
        Document(
            id="3",
            title="Обробка природної мови",
            content="NLP використовує машинне навчання для аналізу "
                   "та розуміння людської мови.",
            keywords={"NLP", "AI", "мова"}
        ),
        Document(
            id="4",
            title="Рекомендаційні системи",
            content="Рекомендаційні системи використовують машинне навчання "
                   "для персоналізації контенту та пропозицій.",
            keywords={"ML", "рекомендації"}
        )
    ]
    
    # Додаємо документи до системи
    logger.info("Додавання документів...")
    for doc in documents:
        search_engine.add_document(doc)
    
    # Будуємо семантичний граф
    logger.info("Побудова семантичного графу...")
    semantic_search.build_semantic_graph(documents)
    
    # Демонстрація різних типів пошуку
    logger.info("\nДемонстрація пошуку:")
    
    # 1. Базовий пошук
    logger.info("\n1. Базовий пошук за 'машинне навчання':")
    results = search_engine.search("машинне навчання")
    for doc in results:
        logger.info(f"- {doc.title}")
    
    # 2. Семантичний пошук
    logger.info("\n2. Знаходження семантичного шляху між документами:")
    path = semantic_search.find_semantic_path("1", "3")
    logger.info(f"Шлях між ML та NLP: {' -> '.join(path)}")
    
    # 3. Рекомендації
    logger.info("\n3. Рекомендації на основі документа про ML:")
    recommendations = semantic_search.recommend_documents("1")
    logger.info(f"Рекомендовані документи: {recommendations}")
    
    # 4. Оптимізований пошук
    logger.info("\n4. Оптимізований пошук:")
    for _ in range(15):  # Симулюємо популярний запит
        search_optimizer.register_query("AI")
    
    optimized_results = search_optimizer.optimize_search("AI")
    logger.info(f"Знайдено {len(optimized_results)} документів через оптимізований пошук")
    
    logger.info("\nДемонстрація завершена!")

if __name__ == "__main__":
    run_advanced_demo()
