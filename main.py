from tfspeech.data import Data
from tfspeech.model.gate import Model
from tfspeech.solver import Solver

d = Data("data/toneclassifier", width=256, batch_size=40, epoch_size=10, optimizer="sgd")
m = Model(width=d.width)
mi = m.compile()
s = Solver(d, mi, learning_rate=0.000001, num_epochs=10, verbose=True)
s.train()
