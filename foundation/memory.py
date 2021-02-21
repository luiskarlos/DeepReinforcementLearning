import random
import pickle

from utils import config
from collections import deque


class Memory:
	def __init__(self, config):
		self.long_term_memory = deque(maxlen=config.MEMORY_SIZE)
		self.short_term_memory = deque(maxlen=config.MEMORY_SIZE)

	def isFull(self) -> bool:
		return len(self.long_term_memory) >= config.MEMORY_SIZE
	
	def commit_stmemory(self, identities) -> 'Memory':
		for r in identities:
			self.short_term_memory.append({
				'board': r[0].board
				, 'state': r[0]
				, 'id': r[0].id
				, 'AV': r[1]
				, 'playerTurn': r[0].playerTurn
				})
		return self

	def commit_ltmemory(self, playerTurn, value) -> 'Memory':
		for move in self.short_term_memory:
			if move['playerTurn'] == playerTurn:
				move['value'] = value
			else:
				move['value'] = -value
			self.long_term_memory.append(move)
			
		self.clear_stmemory()
		return self
	
	def clear_stmemory(self) -> 'Memory':
		self.short_term_memory = deque(maxlen=config.MEMORY_SIZE)
		return self

	def miniBach(self):
		return random.sample(self.long_term_memory, min(config.BATCH_SIZE, len(self.long_term_memory)))

	def sample(self):
		return random.sample(self.long_term_memory, min(config.MEMORY_SAMPLE_SIZE, len(self.long_term_memory)))

	@staticmethod
	def load(path) -> 'Memory':
		return pickle.load(open(path, "rb"))
