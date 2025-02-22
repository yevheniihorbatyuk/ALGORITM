from text_analysis_system import Document, SearchEngine

def main():
    # Ініціалізуємо пошукову систему
    search_engine = SearchEngine()
    
    # Створюємо тестові документи
    documents = [
        Document(
            id="1",
            title="Алгоритми на графах",
            content="BFS та DFS використовуються для обходу графа. "
                   "Ці алгоритми є фундаментальними в теорії графів.",
            keywords={"алгоритми", "графи", "пошук", "bfs", "dfs"}
        ),
        Document(
            id="2",
            title="Структури даних: Префіксне дерево",
            content="Trie (префіксне дерево) - це структура даних для ефективного "
                   "пошуку за префіксом. Часто використовується в автодоповненні.",
            keywords={"структури даних", "дерева", "пошук", "префікс"}
        ),
        Document(
            id="3",
            title="Алгоритм PageRank",
            content="PageRank - алгоритм для оцінки важливості вершин у графі. "
                   "Спочатку використовувався Google для ранжування веб-сторінок.",
            keywords={"алгоритми", "графи", "ранжування", "pagerank"}
        )
    ]
    
    # Додаємо документи до системи
    for doc in documents:
        search_engine.add_document(doc)
        print(f"Додано документ: {doc.title}")
    
    # Демонструємо різні типи пошуку
    print("\n1. Пошук за ключовим словом 'граф':")
    results = search_engine.search("граф")
    for doc in results:
        print(f"- {doc.title}")
    
    print("\n2. Пошук схожих документів до 'Алгоритми на графах':")
    similar_docs = search_engine.document_graph.get_similar_documents("1")
    for doc_id in similar_docs:
        print(f"- Документ {doc_id}")
    
    print("\n3. Автодоповнення для префікса 'алго':")
    docs_with_algo = search_engine.trie.search("алго")
    print(f"Знайдено документів: {len(docs_with_algo)}")
    
    print("\n4. Пошук з використанням wildcard 'граф?':")
    wildcard_results = search_engine.trie.search_with_wildcard("граф?")
    print(f"Знайдено документів з wildcard: {len(wildcard_results)}")

    # Демонстрація роботи з кешем
    print("\n5. Повторний пошук (з кешу):")
    cached_results = search_engine.search("граф")
    for doc in cached_results:
        print(f"- {doc.title} (з кешу)")

if __name__ == "__main__":
    main()