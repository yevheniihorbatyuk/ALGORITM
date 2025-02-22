from typing import Dict, List, Set, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from collections import defaultdict

class TextAnalyzer:
    """Розширений клас для аналізу тексту"""
    
    def __init__(self, text_processor: TextProcessor):
        self.text_processor = text_processor
        self.vectorizer = TfidfVectorizer()
        self.document_vectors = {}
        
    def calculate_document_similarity(self, doc1: Document, doc2: Document) -> float:
        """Обчислює схожість між документами використовуючи TF-IDF"""
        texts = [doc1.content, doc2.content]
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)
    
    def extract_keywords(self, document: Document, top_k: int = 10) -> List[Tuple[str, float]]:
        """Витягує ключові слова з документа на основі TF-IDF"""
        # Отримуємо токени
        tokens = self.text_processor.process(document.content)
        text = " ".join(tokens)
        
        # Обчислюємо TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform([text])
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Отримуємо важливість кожного слова
        word_importance = [(feature_names[i], tfidf_matrix[0, i]) 
                         for i in range(len(feature_names))]
        
        # Сортуємо за важливістю та повертаємо top_k
        word_importance.sort(key=lambda x: x[1], reverse=True)
        return word_importance[:top_k]

class SemanticSearchEngine:
    """Розширення пошукової системи для семантичного пошуку"""
    
    def __init__(self, search_engine: SearchEngine, text_analyzer: TextAnalyzer):
        self.search_engine = search_engine
        self.text_analyzer = text_analyzer
        self.semantic_graph = nx.Graph()
        
    def build_semantic_graph(self, documents: List[Document]):
        """Будує семантичний граф на основі схожості документів"""
        for i, doc1 in enumerate(documents):
            for doc2 in documents[i+1:]:
                similarity = self.text_analyzer.calculate_document_similarity(doc1, doc2)
                if similarity > 0.3:  # Поріг схожості
                    self.semantic_graph.add_edge(doc1.id, doc2.id, weight=similarity)
    
    def find_semantic_path(self, start_doc: str, end_doc: str) -> List[str]:
        """Знаходить семантичний шлях між документами"""
        try:
            path = nx.shortest_path(
                self.semantic_graph, 
                start_doc, 
                end_doc, 
                weight='weight'
            )
            return path
        except nx.NetworkXNoPath:
            return []
    
    def recommend_documents(self, doc_id: str, limit: int = 5) -> List[str]:
        """Рекомендує документи на основі випадкового блукання"""
        if doc_id not in self.semantic_graph:
            return []
        
        # Використовуємо персоналізований PageRank
        personalization = {node: 1.0 if node == doc_id else 0.0 
                         for node in self.semantic_graph.nodes()}
        
        pr = nx.pagerank(
            self.semantic_graph,
            alpha=0.85,
            personalization=personalization
        )
        
        # Сортуємо за рангом та виключаємо вихідний документ
        recommendations = [(node, rank) for node, rank in pr.items() 
                         if node != doc_id]
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return [doc_id for doc_id, _ in recommendations[:limit]]

class SearchOptimizer:
    """Клас для оптимізації пошуку"""
    
    def __init__(self, search_engine: SearchEngine):
        self.search_engine = search_engine
        self.query_patterns = defaultdict(int)
        self.popular_queries = set()
        
    def register_query(self, query: str):
        """Реєструє пошуковий запит для аналізу патернів"""
        self.query_patterns[query] += 1
        if self.query_patterns[query] > 10:  # Поріг популярності
            self.popular_queries.add(query)
    
    def optimize_search(self, query: str) -> List[Document]:
        """Оптимізує пошук на основі популярних запитів"""
        # Якщо запит популярний, спершу шукаємо в кеші
        if query in self.popular_queries:
            cached_results = self.search_engine.cache.get(f"search:{query}")
            if cached_results:
                return json.loads(cached_results)
        
        # Інакше виконуємо звичайний пошук
        results = self.search_engine.search(query)
        
        # Реєструємо запит для майбутньої оптимізації
        self.register_query(query)
        
        return results

def create_demo_system() -> Tuple[SearchEngine, SemanticSearchEngine, SearchOptimizer]:
    """Створює демонстраційну систему з усіма компонентами"""
    text_processor = TextProcessor()
    search_engine = SearchEngine()
    text_analyzer = TextAnalyzer(text_processor)
    semantic_search = SemanticSearchEngine(search_engine, text_analyzer)
    search_optimizer = SearchOptimizer(search_engine)
    
    return search_engine, semantic_search, search_optimizer
