import importlib
import simulation
import data_processing

importlib.reload(simulation)
results = simulation.start()
importlib.reload(data_processing)
data_processing.start(*resultswolff)

