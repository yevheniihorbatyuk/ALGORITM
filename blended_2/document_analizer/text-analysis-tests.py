import unittest
from text_analysis_system import Document, TextProcessor, Trie, DocumentGraph, SearchEngine

class TestTextAnalysisSystem(unittest.TestCase):
    def setUp(self):
        self.text_processor = TextProcessor()
        self.trie = Trie()
        self.document_graph = DocumentGraph()
        self.search_engine = SearchEngine()
        
        # Тестові документи
        self.doc1 = Document(
            id="1",
            title="Алгоритми на графах",
            content="BFS та DFS використовуються для обходу графа",
            keywords={"алгоритми", "графи", "пошук"}
        )
        
        self.doc2 = Document(
            id="2",
            title="Структури даних",
            content="Trie корисний для пошуку за префіксом",
            keywords={"структури даних", "дерева", "пошук"}
        )
    
    def test_text_processor(self):
        tokens = self.text_processor.process("BFS та DFS використовуються для обходу графа")
        self.assertTrue(len(tokens) > 0)
        self.assertFalse(any(token.isspace() for token in tokens))
    
    def test_trie_insert_and_search(self):
        self.trie.insert("графи", "1")
        self.trie.insert("граф", "1")
        self.trie.insert("графіка", "2")
        
        # Тест звичайного пошуку
        results = self.trie.search("граф")
        self.assertEqual(results, {"1", "2"})
        
        # Тест пошуку з wildcard
        results = self.trie.search_with_wildcard("граф?")
        self.assertEqual(results, {"1"})
    
    def test_document_graph(self):
        self.document_graph.add_document(self.doc1)
        self.document_graph.add_document(self.doc2)
        self.document_graph.add_relationship("1", "2", 0.8)
        
        similar_docs = self.document_graph.get_similar_documents("1")
        self.assertEqual(len(similar_docs), 1)
        self.assertEqual(similar_docs[0], "2")
    
    def test_search_engine(self):
        # Додаємо документи до пошукової системи
        self.search_engine.add_document(self.doc1)
        self.search_engine.add_document(self.doc2)
        
        # Тестуємо пошук
        results = self.search_engine.search("граф")
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0].id, "1")

if __name__ == '__main__':
    unittest.main()
