from theanospeech.data import Data
from theanospeech.model import Model
from theanospeech.solver import Solver
import time

d = Data("data/data_linear.json", width=128, batch_size=10, epoch_size=40)

model = Model(width=d.width, num_channels=1, num_classes=4, batch_size=10, conv_settings=[(3, 64)], hidden_dims=[], reg=2e-3)
update, predict = model.compile()
s = Solver(d, update, predict, batch_size=10, learning_rate=1e-4, num_epochs=400, verbose=True)

#model = Model(width=d.width, num_channels=1, num_classes=4, batch_size=10, conv_settings=[], hidden_dims=[], reg=1e-2)
#update, predict = model.compile()
#s = Solver(d, update, predict, batch_size=10, learning_rate=3e-5, num_epochs=2000, verbose=True)

time1 = time.time()
s.train()
time2 = time.time()
print 'time cost:', time2 - time1
print 'best on test:', s.best_test, 'on', s.best_test_t
print 'best on test new:', s.best_test_new, 'on', s.best_test_new_t
