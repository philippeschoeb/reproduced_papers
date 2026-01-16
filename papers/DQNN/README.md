# Distributed Quantum Neural Networks on Distributed Photonic Quantum Computing

The paper reproduced here was one of the projects of the Perceval Challenge: “Distributed Quantum Neural Networks on Distributed Photonic Quantum Computing” by Chen et al., 2025 https://arxiv.org/html/2505.08474v1.

The original code using the Perceval library can be found [here](https://github.com/Louisanity/PhotonicQuantumTrain). 

Current Merlin repo: **TODO**

### 🎯 Main goal 
>Classify the reduced MNIST dataset still containing the full 10 digits

### Main result

>“By leveraging universal linear-optical interferometers ([…]) and matrix product state (MPS) mapping, our framework achieves 10× parameter compression (χ = 4) with only 3.50% relative accuracy loss on MNIST classification (93.29% ± 0.62% vs classical 96.89% ± 0.31%)“


### Main contributions of the paper

> - Compared to traditionnal QML methods, the authors explore a way to use a photonic quantum computer to train the classical parameters of a fully classical neural network model.
> - The main idea driving the project is that fewer quantum parameters need to be trained to obtain all of the classical parameters. They use even less parameters whil having better or equal performance as classical compression techniques such as pruning and weight sharing.
>- After a noise analysis of the brightness, indistinguishability, second-order correlation and transmittance:
>- “Across all sweeps the worst-case degradation is confined to less than three percentage points, identifying excess multi-photon emission at high brightness as the principal residual error source and demonstrating that the hybrid photonic–classical architecture maintains high-fidelity operation under first-order imperfections realistic for current hardware.”

### Their framework

![](images/md1.png)
Source: K.-C. Chen, C.-Y. Liu, Y. Shang, F. Burt, and K. K. Leung, “Distributed Quantum Neural Networks on Distributed Photonic Quantum Computing,” May 13, 2025, arXiv: arXiv:2505.08474. doi: 10.48550/arXiv.2505.08474.


>Here the bond dimension of the MPS, directly impacting its expressivity is an hyperparameter.
>
>The same alternating input state (|01010101⟩) is used for both interferometers.
>
>The input size of the MPS is ⌈log2m⌉+1 where m is the number of classical parameters in the CNN

### Their results

#### Simple Classical CNN
| \# of training parameters | Training accuracy (%) |Testing accuracy (%)|Generalization error |
| ----------- | ----------- |----------- |----------- |
| 6690      | 99.983± 0.02       | 96.890±0,31      | 0.1690±0.005       |

#### Full QT framework varying the bond lenght (χ)
![](images/md2.png)
Source: K.-C. Chen, C.-Y. Liu, Y. Shang, F. Burt, and K. K. Leung, “Distributed Quantum Neural Networks on Distributed Photonic Quantum Computing,” May 13, 2025, arXiv: arXiv:2505.08474. doi: 10.48550/arXiv.2505.08474.

| Bond dimension| \# of training parameters | Training accuracy (%) |Testing accuracy (%)|Generalization error |
| ----------- | ----------- |----------- |----------- |----------- |
| 1     |223      | 58.256 ± 2.34     | 55.775 ± 3.27       | 0.0219 ± 0.007   |
| 2     | 316      | 83.340 ± 2.77     | 81.375 ± 2.28       | 0.0462 ± 0.032   |
| 3     | 471       | 88.693 ± 1.67      | 87.057 ± 2.66       | 0.0364 ± 0.016    |
| 4     | 688       | 93.916 ± 0.45     | 93.292 ± 0.62       | 0.0679 ± 0.002    |
| 5     | 967       | 95.450 ± 0.39     | 93.042 ± 0.77       | 0.0950 ± 0.010    |
| 6     | 1308     | 96.953 ± 0.02     | 94.917 ± 0.60       | 0.1135 ± 0.013    |
| 7     | 1711      | 97.773 ± 0.22     | 94.957 ± 0.82       | 0.1315 ± 0.031   |
| 8     | 2176       | 97.866 ± 0.78      | 94.707 ± 0.4       | 0.1399 ± 0.007   |
| 9     | 2703       | 98.373 ± 0.12      | 94.835 ± 0.4       | 0.1624 ± 0.021    |
| 10     | 3292       | 98.990 ± 0.34     | 95.502 ± 0.84      | 0.2552 ± 0.053    |

#### Comparing QPT with classical compressing methods
![](images/md3.png)
Source: K.-C. Chen, C.-Y. Liu, Y. Shang, F. Burt, and K. K. Leung, “Distributed Quantum Neural Networks on Distributed Photonic Quantum Computing,” May 13, 2025, arXiv: arXiv:2505.08474. doi: 10.48550/arXiv.2505.08474.

|Method| \# of training parameters |Testing accuracy (%)
| ----------- | ----------- |----------- |
| Original      | 6690       | 96.890 ± 0.31      | 
| Weight sharing      | 4770       | 88.666 ± 1.207      | 
| Pruning      | 3770      | 94.443 ± 0.923      | 
| Photonic QT (χ=10)      | 3292       | 95.502 ± 0.84      | 
| Photonic QT (χ=4)     | 688      | 93.292 ± 0.62      | 

#### Ablation analysis
![](images/md4.png)
Source: K.-C. Chen, C.-Y. Liu, Y. Shang, F. Burt, and K. K. Leung, “Distributed Quantum Neural Networks on Distributed Photonic Quantum Computing,” May 13, 2025, arXiv: arXiv:2505.08474. doi: 10.48550/arXiv.2505.08474.

>Here they replaced the quantum layer with a random weight generator as input to the MPS.
>>Altough, after multiple relectures and discussions, it is not clear if at every epoch and training batch, a new random vector is generated making the previous graph not a real ablation analysis.

### Our results

**To come…**