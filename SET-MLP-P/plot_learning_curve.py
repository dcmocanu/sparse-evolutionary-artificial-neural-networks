import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

epsilon=13
samples=2000

set_mlp=np.loadtxt("Pretrained_results/set_mlp_"+str(samples)+"_training_samples_e"+str(epsilon)+"_rand0.txt")
fixprob_mlp=np.loadtxt("Pretrained_results/fixprob_mlp_"+str(samples)+"_training_samples_e"+str(epsilon)+"_rand0.txt")
fc_mlp=np.loadtxt("Pretrained_results/fc_mlp_"+str(samples)+"_training_samples_rand0.txt")

"""
for i in range(1,5):
    set_mlp = set_mlp + np.loadtxt(
        "Results/set_mlp_" + str(samples) + "_training_samples_e" + str(epsilon) + "_rand" + str(i) + ".txt")
    fixprob_mlp = fixprob_mlp + np.loadtxt(
        "Results/fixprob_mlp_" + str(samples) + "_training_samples_e" + str(epsilon) + "_rand" + str(i) + "")
    fc_mlp = fc_mlp + np.loadtxt("Pretrained_results/fc_mlp_" + str(samples) + "_training_samples_rand" + str(i) + ".txt")

set_mlp/=5
fixprob_mlp/=5
fc_mlp/=5
"""
font = { 'size'   : 9}
fig = plt.figure(figsize=(10,5))
matplotlib.rc('font', **font)
fig.subplots_adjust(wspace=0.2,hspace=0.05)

ax1=fig.add_subplot(1,2,1)
ax1.plot(set_mlp[:,2]*100, label="SET-MLP train accuracy", color="r")
ax1.plot(set_mlp[:,3]*100, label="SET-MLP test accuracy", color="b")
ax1.plot(fixprob_mlp[:,2]*100, label="MLP$_{FixProb}$ train accuracy", color="g")
ax1.plot(fixprob_mlp[:,3]*100, label="MLP$_{FixProb}$ test accuracy", color="m")
ax1.plot(fc_mlp[:,2]*100, label="FC-MLP train accuracy", color="y")
ax1.plot(fc_mlp[:,3]*100, label="FC-MLP test accuracy", color="k")
ax1.grid(True)
ax1.set_ylabel("Fashion MNIST\nAccuracy [%]")
ax1.set_xlabel("Epochs [#]")
ax1.legend(loc=4,fontsize=8)

ax2=fig.add_subplot(1,2,2)
ax2.plot(set_mlp[:,0], label="SET-MLP train loss", color="r")
ax2.plot(set_mlp[:,1], label="SET-MLP test loss", color="b")
ax2.plot(fixprob_mlp[:,0], label="MLP$_{FixProb}$ train loss", color="g")
ax2.plot(fixprob_mlp[:,1], label="MLP$_{FixProb}$ test loss", color="m")
ax2.plot(fc_mlp[:,0], label="FC-MLP train loss", color="y")
ax2.plot(fc_mlp[:,1], label="FC-MLP test loss", color="k")
ax2.grid(True)
ax2.set_ylabel("Loss (MSE)")
ax2.set_xlabel("Epochs [#]")
ax2.legend(loc=1,fontsize=8)


plt.savefig("Pretrained_results/mnist_learning_curves_samples"+str(samples)+".pdf", bbox_inches='tight')

plt.close()