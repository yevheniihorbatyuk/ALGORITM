import mmh3  # Пакет для хешування
from bitarray import bitarray  # Ефективний бітовий масив

class BloomFilter:
    def __init__(self, size, hash_count):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)

    def add(self, item):
        for i in range(self.hash_count):
            index = mmh3.hash(item, i) % self.size
            self.bit_array[index] = 1

    def check(self, item):
        for i in range(self.hash_count):
            index = mmh3.hash(item, i) % self.size
            if not self.bit_array[index]:
                return False
        return True

if __name__=='__main__':
    # Тестування
    bloom = BloomFilter(1000, 5)
    bloom.add("apple")
    bloom.add("banana")

    print("apple in filter?", bloom.check("apple"))  # True
    print("grape in filter?", bloom.check("grape"))  # False (можливо True через false positive)