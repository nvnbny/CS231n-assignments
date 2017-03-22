import numpy as np

results_lr = {}
results_lr[1.09] = .4
results_lr[1.19] = .4
results_lr[ 1.3712954244581913e-09] = .4

for lr in sorted(results_lr):
    val_accuracy = results_lr[lr]
    print 'lr %e val accuracy: %f' % (lr, val_accuracy)


