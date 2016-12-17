from tfspeech.data import Data
from tfspeech.model.cnn import Model
from tfspeech.solver import Solver

d = Data("data/toneclassifier", width=256, batch_size=5, epoch_size=80, optimizer="sgd")
m = Model(width=d.width, num_channels=2, num_classes=4, conv_settings=[], hidden_dims=[], reg=0.0)
mi = m.compile()
s = Solver(d, mi, learning_rate=0.0000000001, num_epochs=100, verbose=True)
s.train()
