from pre_process import *

# pre_processing = PreProcessing('../data/pokemon.csv')
pre_processing = PreProcessing('../data/train.csv')
pre_processing.load_dataset()
pre_processing.translate_columns()
# pre_processing.save_file('../data/pokemon.csv')
pre_processing.save_file('../data/train.csv')