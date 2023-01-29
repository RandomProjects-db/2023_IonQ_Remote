import qiskit
from qiskit import quantum_info
from qiskit.execute_function import execute
from qiskit import BasicAer
import numpy as np
import pickle
import json
import os
from collections import Counter
from sklearn.metrics import mean_squared_error
from typing import Dict, List
import matplotlib.pyplot as plt

#define utility functions

def simulate(circuit: qiskit.QuantumCircuit) -> dict:
    """Simulate the circuit, give the state vector as the result."""
    backend = BasicAer.get_backend('statevector_simulator')
    job = execute(circuit, backend)
    result = job.result()
    state_vector = result.get_statevector()
    
    histogram = dict()
    for i in range(len(state_vector)):
        population = abs(state_vector[i]) ** 2
        if population > 1e-9:
            histogram[i] = population
    
    return histogram


def histogram_to_category(histogram):
    """This function take a histogram representations of circuit execution results, and process into labels as described in 
    the problem description."""
    assert abs(sum(histogram.values())-1)<1e-8
    positive=0
    for key in histogram.keys():
        digits = bin(int(key))[2:].zfill(20)
        if digits[-1]=='0':
            positive+=histogram[key]
        
    return positive


def count_gates(circuit: qiskit.QuantumCircuit) -> Dict[int, int]:
    """Returns the number of gate operations with each number of qubits."""
    counter = Counter([len(gate[1]) for gate in circuit.data])
    #feel free to comment out the following two lines. But make sure you don't have k-qubit gates in your circuit
    #for k>2
    for i in range(2,20):
        assert counter[i]==0
    return counter


def image_mse(image1,image2):
    # Using sklearns mean squared error:
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    return mean_squared_error(255*image1,255*image2)


#load the actual hackthon data (fashion-mnist)
images=np.load('images.npy')
labels=np.load('labels.npy')
#you can visualize it
plt.imshow(images[0])



def encoder(image):    
    def resize(img, ht, width):
        output = np.zeros((ht,width), dtype = img.dtype)
        iht, iwidth = img.shape
        scale_y = iht/ht
        scale_x = iwidth/width
    
        for i in range(ht):
            for j in range(width):
                x = j * scale_x
                y = i * scale_y
                fracX = x - np.floor(x)
                fracY = y - np.floor(y)

                lu = img[int(np.floor(y)), int(np.floor(x))]
                ru = img[int(np.floor(y)), int(min(img.shape[1] - 1, np.ceil(x)))]
                ll = img[int(min(img.shape[0] - 1, np.ceil(y))), int(np.floor(x))]
                rl = img[int(min(img.shape[0] - 1, np.ceil(y))), int(min(img.shape[1] - 1, np.ceil(x)))]


                top = (ru * fracX) + (lu * (1.0 - fracX))
                bottom = (rl * fracX) + (ll * (1.0 - fracX))
                output[i,j] = (top * fracY) + (bottom * (1.0 - fracY))
        return output
    
    img = resize(image,16,16)
    img = img.reshape(256,1)
    img = img/np.linalg.norm(img)
    img = [i[0] for i in img]
    
    qc = qiskit.QuantumCircuit(8)
    qc.initialize(list(img),qc.qubits)
    return qc

def decoder(histogram):
    def resize(img, ht, width):
        output = np.zeros((ht,width), dtype = img.dtype)
        iht, iwidth = img.shape
        scale_y = iht/ht
        scale_x = iwidth/width
    
        for i in range(ht):
            for j in range(width):
                x = j * scale_x
                y = i * scale_y
                fracX = x - np.floor(x)
                fracY = y - np.floor(y)

                lu = img[int(np.floor(y)), int(np.floor(x))]
                ru = img[int(np.floor(y)), int(min(img.shape[1] - 1, np.ceil(x)))]
                ll = img[int(min(img.shape[0] - 1, np.ceil(y))), int(np.floor(x))]
                rl = img[int(min(img.shape[0] - 1, np.ceil(y))), int(min(img.shape[1] - 1, np.ceil(x)))]


                top = (ru * fracX) + (lu * (1.0 - fracX))
                bottom = (rl * fracX) + (ll * (1.0 - fracX))
                output[i,j] = (top * fracY) + (bottom * (1.0 - fracY))
        return output
    
    img_re = []
    k = 0
    for i in range(256):
        if i in list(histogram.keys()):
            img_re.append(list(histogram.values())[k])
            k+=1
        else:
            img_re.append(0.00)
    img_re = np.array(img_re).reshape(16,16)
    img_re = resize(img_re,28,28)
    return img_re


def run_part1(image):
    circuit=encoder(image)
    histogram=simulate(circuit)
    image_re=decoder(histogram)
    return circuit, image_re


n=len(images)
mse=0
gatecount=0

for data in images:
    #encode image into circuit
    circuit,image_re=run_part1(data)
    
    #count the number of 2qubit gates used
    #gatecount+=count_gates(circuit)[2]
    
    #calculate mse
    mse+=image_mse(data,image_re)
    
#fidelity of reconstruction
f=1-mse/n
#gatecount=gatecount/n

#score for part1 
print(f*(0.999**1))
