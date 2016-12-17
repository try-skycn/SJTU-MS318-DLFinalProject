from tfspeech.data import Data
from tfspeech.model.cnn import Model
from tfspeech.solver import Solver

d = Data("data/toneclassifier", width=256, batch_size=40, epoch_size=10, optimizer="adam")
m = Model(width=d.width, num_channels=2, num_classes=4, conv_settings=[(5, 16), (5, 16), (5, 16)], hidden_dims=[256, 256, 256])
mi = m.compile()
s = Solver(d, mi, learning_rate=5, num_epochs=100, verbose=True)
s.train()
