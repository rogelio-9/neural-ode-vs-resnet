# Neural ODE vs. ResNet: An Empirical Comparison on Image Classification

> **CS 5365: Deep Learning - Final Project**
> Rogelio Lozano · Joe Mota
> Department of Computer Science, The University of Texas at El Paso

---

## Introduction

This repository contains a from-scratch re-implementation of the architectures proposed in:

> **Chen, R.T.Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018).**
> *Neural Ordinary Differential Equations.* NeurIPS 2018.
> [[arXiv:1806.07366]](https://arxiv.org/abs/1806.07366)

The paper introduces **Neural ODEs** - a family of models that replace discrete residual layers with a continuous-depth dynamical system governed by an ordinary differential equation:

$$\frac{dh(t)}{dt} = f(h(t), t, \theta)$$

The hidden state evolves continuously from $t=0$ to $t=1$, with a standard ODE solver handling integration. Two key advantages are claimed: (1) the solver **adaptively** determines computational depth per input, and (2) the **adjoint sensitivity method** enables backpropagation in constant memory, independent of solver steps. The paper's central claim is that Neural ODEs can match ResNet-level accuracy while using significantly fewer parameters and less GPU memory.

We re-implement both a ResNet baseline (`SmallResNet`) and a Neural ODE classifier (`NeuralODENet`) and evaluate them on **MNIST** and **CIFAR-10** to assess how well this claim holds in practice.

---

## Chosen Result

We targeted **Table 1** from Chen et al. (2018), which reports that a Neural ODE classifier matches a comparable ResNet on MNIST-scale tasks while using a fraction of the parameters and achieving lower peak memory through the adjoint method.

This result matters because it is the **central empirical justification** for Neural ODEs as a practical replacement for ResNets in memory-constrained settings — not just a theoretical curiosity. If it holds, Neural ODEs become a compelling option wherever memory is a bottleneck.

| Dataset   | Model      | Params  | Test Acc. | Train Time | Peak Mem. |
|-----------|------------|---------|-----------|------------|-----------|
| MNIST     | ResNet     | 174,970 | **99.53%**| 120.5s     | 145 MB    |
| MNIST     | Neural ODE | 52,138  | 96.85%    | 946.7s     | 121 MB    |
| CIFAR-10  | ResNet     | 696,618 | **90.84%**| 550.8s     | 520 MB    |
| CIFAR-10  | Neural ODE | 207,818 | 65.44%    | 2,962.6s   | 290 MB    |

*Both Neural ODE models use ~3.4× fewer parameters and significantly less GPU memory than their ResNet counterparts.*

---

## Repository Contents

```
neural-ode-vs-resnet/
├── README.md                  ← You are here
├── code/
│   └── Neural_ODE_vs_ResNet.ipynb   ← Full training & evaluation notebook
├── data/
│   └── README.md              ← Instructions for obtaining MNIST & CIFAR-10
├── results/                   ← Generated figures, CSVs, and logs
├── poster/
│   └── poster.pdf             ← In-class presentation poster
├── report/
│   └── report.pdf             ← 2-page project summary report
├── LICENSE                    ← MIT License
└── .gitignore
```

---

## Re-implementation Details

### Architectures

**ResNet (`SmallResNet`)** - A 3-stage residual network following the standard `conv → BN → ReLU → conv → BN` pattern with skip connections, global average pooling, and a fully connected classifier head. Base channel width is tunable to match parameter counts against the Neural ODE.

**Neural ODE (`NeuralODENet`)** — Mirrors the paper's image classifier design:
1. **Downsampling stem** — CNN layers reduce spatial resolution and lift channel count
2. **ODEBlock** — Wraps `odeint_adjoint` (torchdiffeq) to solve continuous dynamics in feature space
3. **Classifier head** — Global average pool + fully connected layer

The ODE dynamics function `f(t, h)` is a small 2-layer convolutional network. We used the `dopri5` adaptive Runge-Kutta solver with `rtol = atol = 1e-2`, matching the paper's default setup.

### Datasets

| Dataset  | Images    | Input Shape | Classes | Split           |
|----------|-----------|-------------|---------|-----------------|
| MNIST    | 70,000    | 1×28×28     | 10      | 55k / 5k / 10k  |
| CIFAR-10 | 60,000    | 3×32×32     | 10      | 45k / 5k / 10k  |

CIFAR-10 training used standard light augmentation (random crop + horizontal flip). All splits are seeded with `SEED=42` for full reproducibility.

### Training Configuration

| Parameter     | Value              |
|---------------|--------------------|
| Optimizer     | Adam               |
| Learning rate | 1e-3               |
| LR schedule   | Cosine annealing   |
| MNIST epochs  | 20                 |
| CIFAR epochs  | 60                 |
| Batch size    | 128                |
| ODE solver    | dopri5 (adaptive)  |
| rtol / atol   | 1e-2 / 1e-2        |
| Backprop      | Adjoint method     |

### Evaluation Metrics

- **Test accuracy** (top-1)
- **Total training time** (wall clock, seconds)
- **Peak GPU memory** (MB, via `torch.cuda.max_memory_allocated`)
- **Number of Function Evaluations (NFE)** per batch — key Neural ODE metric tracking adaptive solver steps

### Challenges & Modifications

- **CIFAR-10 instability**: The Neural ODE's val accuracy oscillated wildly between epochs on CIFAR-10 (dropping from ~63% to ~10%), likely due to the adaptive solver struggling to maintain a stable trajectory as the loss landscape shifted. This instability was absent on MNIST.
- **Solver tolerance tradeoff**: `rtol = atol = 1e-2` was chosen to balance speed and accuracy. Tighter tolerances would likely improve stability but at a prohibitive compute cost (CIFAR-10 already took ~50 minutes).
- **Representational power gap**: Our ODE function operates in a fixed 64-channel feature space, while the ResNet has three progressively wider stages — this likely explains most of the 25-point CIFAR-10 accuracy gap.
- **Parameter matching**: We tuned `base_ch` (ResNet) and `ch` (Neural ODE) so the Neural ODE had ~3× fewer parameters on both datasets, consistent with the paper's setup.

---

## Reproduction Steps

### Prerequisites

```bash
# Python 3.9+ recommended
pip install torch torchvision torchdiffeq numpy pandas matplotlib tqdm
```

> **GPU strongly recommended.** Training was done on an NVIDIA A100 (40GB VRAM) via Google Colab. With a consumer GPU (e.g., RTX 3080), expect roughly similar MNIST times but significantly longer CIFAR-10 runs. CPU-only training is not recommended for CIFAR-10.

### Option A - Google Colab (Recommended)

1. Open `code/Neural_ODE_vs_ResNet.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Select **Runtime → Change runtime type → GPU (A100 or T4)**
3. Run all cells top to bottom — datasets download automatically

### Option B - Local Jupyter

```bash
git clone https://github.com/<your-username>/neural-ode-vs-resnet.git
cd neural-ode-vs-resnet
pip install torch torchvision torchdiffeq numpy pandas matplotlib tqdm jupyter
jupyter notebook code/Neural_ODE_vs_ResNet.ipynb
```

### Key Configuration

All hyperparameters live in the `CFG` dict at the top of the notebook:

```python
CFG = {
    "seed":         42,
    "batch_size":   128,
    "mnist_epochs": 20,
    "cifar_epochs": 60,
    "ode_method":   "dopri5",
    "ode_rtol":     1e-2,
    "ode_atol":     1e-2,
    "adjoint":      True,
}
```

### Expected Runtime

| Experiment          | A100 (Colab) | RTX 3080 (est.) |
|---------------------|-------------|-----------------|
| ResNet - MNIST      | ~2 min      | ~3–5 min        |
| Neural ODE - MNIST  | ~16 min     | ~30–45 min      |
| ResNet - CIFAR-10   | ~9 min      | ~15–20 min      |
| Neural ODE - CIFAR-10 | ~49 min   | ~90–120 min     |

### Obtaining the Data

MNIST and CIFAR-10 are downloaded automatically via `torchvision.datasets` on first run. See `data/README.md` for details and manual download instructions.

---

## Results & Insights

### What to Expect After Running

Running the full notebook produces four trained models and prints a results table for each. Generated figures (loss curves, NFE curves, efficiency bar charts) are saved to `results/`.

### MNIST - Claim Largely Confirmed

The Neural ODE reached **96.85%** test accuracy vs. **99.53%** for ResNet — only 2.7 points lower — with **3.4× fewer parameters** and **16% less peak GPU memory**. The memory efficiency claim from the paper holds clearly. The main cost is training time: ~16 min vs. ~2 min due to ODE solver overhead.

### CIFAR-10 - Significant Divergence

The Neural ODE only reached **65.44%** vs. **90.84%** for ResNet - a 25-point gap not implied by the original paper. The adaptive solver averaged ~75 NFE/batch on CIFAR-10 (vs. ~60 on MNIST), and val accuracy oscillated severely throughout training. Despite this, the **memory advantage held**: 290 MB vs. 520 MB (44% less), with 3.4× fewer parameters.

### Key Takeaway

> **Neural ODEs are a compelling ResNet alternative when memory and parameter efficiency matter more than accuracy on complex tasks.** On simple datasets they are competitive; on harder distributions, ODE solving overhead and training instability become real barriers that the original paper understandably glosses over.

---

## Conclusion

This project gave us a much deeper appreciation for what Neural ODEs actually do versus what they sound like on paper. The continuous-depth framing is elegant and the memory efficiency is real and reproducible - but matching ResNet accuracy on anything beyond simple benchmarks requires careful architecture choices and training stability work. The adaptive solver introduces a feedback loop between the loss landscape and compute steps taken, making Neural ODEs fundamentally harder to tune than a ResNet where every epoch is structurally identical.

**Key lessons learned:**
- Neural ODE training stability is highly sensitive to solver tolerances — a tradeoff the paper does not surface
- The ODE's representational power in feature space is the bottleneck, not the continuous-depth formulation itself
- Memory efficiency is real and consistent; accuracy parity is dataset-dependent

**Natural next steps** would be scaling the ODE feature extractor, testing fixed-step solvers (Euler, RK4) for better speed/accuracy control, exploring hybrid ResNet+ODE architectures, and benchmarking on CIFAR-100 or TinyImageNet.

---

## References

1. R.T.Q. Chen, Y. Rubanova, J. Bettencourt, D. Duvenaud. *Neural Ordinary Differential Equations.* NeurIPS 2018. [[arXiv:1806.07366]](https://arxiv.org/abs/1806.07366)
2. K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* CVPR 2016.
3. R.T.Q. Chen. *torchdiffeq: Differentiable ODE solvers with full GPU support and adjoint backpropagation.* GitHub, 2018. [[repo]](https://github.com/rtqichen/torchdiffeq)
4. A. Paszke et al. *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS 2019.
5. Y. LeCun, C. Cortes, C.J. Burges. *The MNIST Database of Handwritten Digits.* 1998. [[link]](http://yann.lecun.com/exdb/mnist/)
6. A. Krizhevsky. *Learning Multiple Layers of Features from Tiny Images.* University of Toronto, 2009.
7. S. Ioffe, C. Szegedy. *Batch Normalization: Accelerating Deep Network Training.* ICML 2015.
8. D.P. Kingma, J. Ba. *Adam: A Method for Stochastic Optimization.* ICLR 2015.

---

## Acknowledgements

Developed for **CS 5365: Deep Learning**, The University of Texas at El Paso, Spring 2026.
Instructor: Prof. Nan Jiang.
Training conducted on Google Colab using an NVIDIA A100 GPU provided by Google.
We thank the open-source communities behind PyTorch, torchdiffeq, and torchvision, and the authors of Chen et al. (2018) for their foundational work on Neural Ordinary Differential Equations.
