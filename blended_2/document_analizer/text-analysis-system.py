from typing import Dict, Set, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import spacy
import networkx as nx
from elasticsearch import Elasticsearch
import redis
import json

@dataclass
class Document:
    """Представляє документ у системі"""
    id: str
    title: str
    content: str
    keywords: Set[str] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = set()

class TextProcessor:
    """Відповідає за обробку тексту: токенізацію, лематизацію, видалення стоп-слів"""
    
    def __init__(self):
        self.nlp = spacy.load("uk_core_news_sm")  # для української мови
        
    def process(self, text: str) -> List[str]:
        """
        Обробляє текст та повертає список токенів
        """
        doc = self.nlp(text)
        # Видаляємо стоп-слова та пунктуацію, лематизуємо
        tokens = [token.lemma_.lower() for token in doc 
                 if not token.is_stop and not token.is_punct]
        return tokens

class TrieNode:
    """Вузол префіксного дерева"""
    
    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.is_end: bool = False
        self.documents: Set[str] = set()

class Trie:
    """Префіксне дерево для швидкого пошуку та автодоповнення"""
    
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str, doc_id: str) -> None:
        """Додає слово до дерева з прив'язкою до документа"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.documents.add(doc_id)
        node.is_end = True
    
    def search(self, prefix: str) -> Set[str]:
        """Шукає всі документи, що містять слова з даним префіксом"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return set()
            node = node.children[char]
        return node.documents
    
    def search_with_wildcard(self, pattern: str) -> Set[str]:
        """Пошук з використанням wildcard символу '?'"""
        def search_recursive(node: TrieNode, pattern: str, current_pos: int) -> Set[str]:
            if current_pos == len(pattern):
                return node.documents if node.is_end else set()
            
            results = set()
            current_char = pattern[current_pos]
            
            if current_char == '?':
                # Для wildcard символу перевіряємо всі можливі символи
                for child in node.children.values():
                    results.update(search_recursive(child, pattern, current_pos + 1))
            elif current_char in node.children:
                results.update(search_recursive(node.children[current_char], 
                                             pattern, current_pos + 1))
            
            return results
        
        return search_recursive(self.root, pattern, 0)

class DocumentGraph:
    """Граф документів та їх зв'язків"""
    
    def __init__(self):
        self.graph = nx.Graph()
    
    def add_document(self, doc: Document) -> None:
        """Додає документ до графу"""
        self.graph.add_node(doc.id, title=doc.title)
    
    def add_relationship(self, doc1_id: str, doc2_id: str, weight: float) -> None:
        """Додає зв'язок між документами з вагою"""
        self.graph.add_edge(doc1_id, doc2_id, weight=weight)
    
    def get_similar_documents(self, doc_id: str, limit: int = 5) -> List[str]:
        """Знаходить найбільш схожі документи"""
        if doc_id not in self.graph:
            return []
        
        # Використовуємо PageRank для знаходження важливих документів
        pr = nx.pagerank(self.graph)
        
        # Знаходимо сусідні документи та сортуємо за вагою
        neighbors = [(neighbor, pr[neighbor]) 
                    for neighbor in self.graph.neighbors(doc_id)]
        neighbors.sort(key=lambda x: x[1], reverse=True)
        
        return [doc_id for doc_id, _ in neighbors[:limit]]

class SearchEngine:
    """Головний клас пошукової системи"""
    
    def __init__(self):
        self.text_processor = TextProcessor()
        self.trie = Trie()
        self.document_graph = DocumentGraph()
        self.es = Elasticsearch(hosts=["localhost:9200"])
        self.cache = redis.Redis(host='localhost', port=6379, db=0)
        
    def add_document(self, document: Document) -> None:
        """Додає документ до всіх компонентів системи"""
        # Обробка тексту
        tokens = self.text_processor.process(document.content)
        
        # Додавання до Trie
        for token in tokens:
            self.trie.insert(token, document.id)
        
        # Додавання до графу
        self.document_graph.add_document(document)
        
        # Додавання до Elasticsearch
        self.es.index(
            index="documents",
            id=document.id,
            document={
                "title": document.title,
                "content": document.content,
                "keywords": list(document.keywords)
            }
        )
    
    def search(self, query: str) -> List[Document]:
        """Виконує пошук документів за запитом"""
        # Спершу перевіряємо кеш
        cache_key = f"search:{query}"
        cached_result = self.cache.get(cache_key)
        
        if cached_result:
            return json.loads(cached_result)
        
        # Пошук в Elasticsearch
        es_result = self.es.search(
            index="documents",
            body={
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^2", "content", "keywords^1.5"]
                    }
                }
            }
        )
        
        # Обробка результатів
        results = []
        for hit in es_result["hits"]["hits"]:
            doc_data = hit["_source"]
            doc = Document(
                id=hit["_id"],
                title=doc_data["title"],
                content=doc_data["content"],
                keywords=set(doc_data["keywords"])
            )
            results.append(doc)
        
        # Зберігаємо в кеші на 1 годину
        self.cache.setex(
            cache_key,
            3600,
            json.dumps([doc.__dict__ for doc in results])
        )
        
        return results
