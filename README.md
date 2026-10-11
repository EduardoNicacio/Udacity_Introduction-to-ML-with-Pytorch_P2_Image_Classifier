# CIFAR-10 Image Classifier

> A PyTorch convolutional neural network that classifies CIFAR-10 images at **84.92% test accuracy** — built to answer one question for a self-driving car startup: *build an image classifier in-house, or buy one from Detectocorp?*

**Verdict: build.** Our in-house model beats Detectocorp's claimed 70% by nearly 15 points, and it was trained in ~12 minutes on a single laptop GPU with only **1.15M parameters**.

---

## Results

| Benchmark | Accuracy | Outcome |
| :--- | ---: | :--- |
| **In-house CNN (this project)** | **84.92%** | ✅ Recommended build |
| Detectocorp vendor claim | 70.00% | ❌ Beaten by +14.9 pts |
| Deep Belief Networks — Krizhevsky, 2010 | 78.90% | Beaten |
| Maxout Networks — Goodfellow et al., 2013 | 90.60% | Headroom remains |
| Wide Residual Networks — Zagoruyko et al., 2016 | 96.00% | Headroom remains |
| GPipe — Huang et al., 2018 | 99.00% | Headroom remains |

### Per-class test accuracy

| Class | plane | car | bird | cat | deer | dog | frog | horse | ship | truck |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Accuracy | 86.2% | **93.8%** | 76.5% | 68.4% | 84.6% | 79.2% | 89.5% | 87.4% | 93.3% | 90.3% |

Training loss fell from **1.629 → 0.420** over 30 epochs; the full log, loss plot, and per-class table are embedded in the notebook. Inference outputs are softmax probabilities, verified to sum to **1.000000** across all 10,000 test images.

---

## Architecture

```txt
Input 3×32×32
  │
  ├─ Conv2d(3→32) → BatchNorm2d → ReLU → MaxPool2d     ─┐
  ├─ Conv2d(32→64) → BatchNorm2d → ReLU → MaxPool2d    ─┤ 3 blocks
  ├─ Conv2d(64→128) → BatchNorm2d → ReLU → MaxPool2d   ─┘ 32×32 → 4×4
  │
  ├─ Flatten (2048)
  ├─ Linear(2048→512) → ReLU → Dropout(p=0.5)
  └─ Linear(512→10) → softmax probabilities (default)
                      return_logits=True → raw logits for CrossEntropyLoss
```

| Choice | Setting |
| :--- | :--- |
| Data augmentation | `RandomCrop(32, padding=4)`, `RandomHorizontalFlip` (train only) |
| Normalization | CIFAR-10 mean `(0.4914, 0.4822, 0.4465)` / std `(0.2470, 0.2435, 0.2616)` |
| Outputs | `forward()` returns 10-class softmax probabilities; training explicitly requests raw logits (`return_logits=True`) |
| Loss | `CrossEntropyLoss` on raw logits |
| Optimizer | SGD — `lr=0.01`, `momentum=0.9`, `weight_decay=5e-4` |
| Scheduler | `CosineAnnealingLR(T_max=30)` |
| Training | 30 epochs, batch size 64 |
| Parameters | 1,147,914 |

---

## Getting Started

**Requirements:** Python 3.8+, PyTorch ≥ 1.4, torchvision ≥ 0.8 (developed on Python 3.14.7, torch 2.14.1+cu132 — CUDA optional; the notebook falls back to CPU automatically).

```bash
pip install torch torchvision matplotlib numpy jupyter
jupyter notebook CIFAR-10_Image_Classifier-STARTER.ipynb
```

Run it headless (downloads CIFAR-10 automatically on first run, ~170 MB):

```bash
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 \
  CIFAR-10_Image_Classifier-STARTER.ipynb
```

### Loading the saved weights

```python
# after the Net class is defined (see the notebook)
net = Net()
net.load_state_dict(torch.load("cifar10_model.pth", map_location="cpu"))
net.eval()
```

---

## Project Structure

```txt
.
├── CIFAR-10_Image_Classifier-STARTER.ipynb   # main deliverable — executed, 0 errors
├── cifar10_model.pth                         # trained weights (state_dict)
├── data/                                     # CIFAR-10 (auto-downloaded)
├── docs/
│   ├── 1. Project Overview.md                # business scenario
│   ├── 2. Working Locally.md                 # environment setup
│   ├── 3. Rubric.md                          # grading criteria
│   └── 4. Candidate Notes.md                 # engineering write-up
└── LICENSE.md                                # MIT
```

---

## Recommendation (tl;dr)

Building in-house is the right call: we already outscore the vendor on their own benchmark, we own every weight (retrainable on proprietary dashcam data via transfer learning), and the compute cost is negligible. The full analysis is in the notebook's **Make a Recommendation** section; the engineering rationale is in [`docs/4. Candidate Notes.md`](docs/4.%20Candidate%20Notes.md).

---

## License

MIT — see [LICENSE.md](LICENSE.md).
