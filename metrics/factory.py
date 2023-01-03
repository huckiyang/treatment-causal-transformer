import importlib

from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger

from .eval import Evaluator

def factory(engine=None, mode=None):

	name = Options()['model.metric.name']
	if name == 'eval':
		metric = Evaluator()
	else:
		raise ValueError(name)

	return metric