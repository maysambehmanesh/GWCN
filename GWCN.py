# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 08:52:35 2021

@author: Maysam
"""

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from spektral.data.loaders import SingleLoader
from spektral.datasets.citation import Citation
from spektral.transforms import AdjToSpTensor, LayerPreprocess
import numpy as np



from GWCN_Net import AGWConv
from GWCN_Func import computeLoaderCombine,TimeHistory
time_callback = TimeHistory()

# Load data
dataset = Citation("cora", transforms=[LayerPreprocess(AGWConv), AdjToSpTensor()])
mask_tr, mask_va, mask_te = dataset.mask_tr, dataset.mask_va, dataset.mask_te

# Parameters
channels = 16  # Number of features in the first layer
iterations =1  # Number of layers
share_weights = False  # Share weights 
dropout_skip = 0.75  # Dropout rate for the internal skip connection 
dropout = 0.25  # Dropout rate for the features
l2_reg = 5e-4  # L2 regularization rate
learning_rate = 1e-2  # Learning rate
epochs = 20000  # Number of training epochs
patience = 100  # Patience for early stopping
a_dtype = dataset[0].a.dtype  # Only needed for TF 2.1

N = dataset.n_nodes  # Number of nodes in the graph
F = dataset.n_node_features  # Original size of node features
n_out = dataset.n_labels  # Number of classes

# AGWC parameters
thr=1e-4    # threshold parameter for check sparsity of wavelet
scales=[0.4,0.9]      # range of scales 
N_scales = len(scales)
m=40                #Order of polynomial approximation
apx_phsi= False     # approximate Phsi


# Model definition
x_in = Input(shape=(F,))
a_in = Input((N,), sparse=True, dtype=a_dtype)
phsi_in = Input(shape=(N,N,), dtype=a_dtype)
phsiIn_in = Input(shape=(N,N,), dtype=a_dtype)




gc_1 = AGWConv(
    channels,
    iterations=iterations,
    order=N_scales,
    share_weights=share_weights,
    dropout_rate=dropout_skip,
    activation="elu",
    gcn_activation="elu",
    kernel_regularizer=l2(l2_reg),
)([x_in, phsi_in, phsiIn_in, a_in])
gc_2 = Dropout(dropout)(gc_1)
gc_2 = AGWConv(
    n_out,
    iterations=1,
    order=1,
    share_weights=share_weights,
    dropout_rate=dropout_skip,
    activation="softmax",
    gcn_activation=None,
    kernel_regularizer=l2(l2_reg),
)([gc_2, phsi_in, phsiIn_in, a_in])


# Build model
model = Model(inputs=[x_in, phsi_in, phsiIn_in, a_in], outputs=gc_2)
optimizer = Adam(learning_rate=learning_rate)
model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", weighted_metrics=["acc"]
)
model.summary()

# Train model
loader_tr = SingleLoader(dataset, sample_weights=mask_tr)
loader_va = SingleLoader(dataset, sample_weights=mask_va)
loader_te = SingleLoader(dataset, sample_weights=mask_te)


loader_tr_load,loader_va_load,loader_te_load=computeLoaderCombine(
    loader_tr,loader_va,loader_te,apx_phsi,N_scales,scales,m,epochs,thr)



print("Training model.")
history=model.fit(
    loader_tr_load,
    steps_per_epoch=loader_tr.steps_per_epoch,
    validation_data=loader_va_load,
    validation_steps=loader_va.steps_per_epoch,
    epochs=epochs,
    callbacks=[EarlyStopping(patience=patience, restore_best_weights=True),time_callback],
)

times = time_callback.times

model.summary()

# Evaluate model
print("Evaluating model.")


eval_results = model.evaluate(loader_te_load, steps=loader_te.steps_per_epoch)
print("Done.\n" "Test loss: {}\n" "Test accuracy: {}".format(*eval_results))

print("\n""Avg. Time/epoch: {}\n" .format(np.average(times)))

# import stellargraph as sg
# sg.utils.plot_history(history)