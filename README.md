# Reliable AI: Adversarial Attack Robustness

이 저장소는 MNIST와 CIFAR-10 데이터셋을 사용하여 CNN 모델을 학습하고, 다양한 적대적 공격(Adversarial Attacks)에 대한 공격 성공률을 확인하는 프로젝트입니다.

## 📋 Requirements

이 프로젝트를 실행하기 위해 필요한 라이브러리는 `requirements.txt`에 명시되어 있습니다. 아래 명령어로 설치할 수 있습니다:

```bash
pip install -r requirements.txt
```

**주요 라이브러리:**
- torch
- torchvision
- tqdm
- argparse

## 🏃 QuickStart

`run.sh` 스크립트를 사용하여 CIFAR10과 MNIST 데이터셋에서의 여러 공격 강도(epsilon: 0.05, 0.1, 0.2, 0.3)에 대한 평가를 한 번에 실행할 수 있습니다.

```bash
bash run.sh
```

## 🚀 Usage

`test.py` 스크립트를 사용하여 모델을 학습하고 평가할 수 있습니다.

### 1. 기본 실행 (Training & Evaluation)

데이터셋을 지정하여 일반적인 학습 및 테스트를 진행합니다.

**MNIST 데이터셋:**
```bash
python test.py --dataset MNIST
```

**CIFAR-10 데이터셋:**
```bash
python test.py --dataset CIFAR10
```

### 2. 적대적 공격 테스트 (Adversarial Attacks)

`--use_attack` 플래그를 사용하면 학습된 모델에 대해 다음의 **모든 적대적 공격**을 수행하고 성능을 평가합니다.

**공격 실행 예시:**
```bash
# 기본 공격 강도(epsilon=0.3)로 테스트
python test.py --dataset MNIST --use_attack

# 공격 강도(epsilon) 조절 (예: 0.1)
python test.py --dataset CIFAR10 --use_attack --attack_eps 0.1
```

## 🛡️ Supported Attacks

이 프로젝트는 `adversarial_attack.py`에 정의된 다음 공격 기법들을 지원합니다:

1.  **FGSM Untargeted** (Fast Gradient Sign Method)
2.  **FGSM Targeted**
3.  **PGD Untargeted** (Projected Gradient Descent)
4.  **PGD Targeted**

## 📊 Outputs

- 터미널에는 정답을 맞춘 이미지들을 기준으로 **공격 성공률(Attack Success Rate, ASR)** 이 디테일하게 출력됩니다.
- 평가 결과(Epoch, Clean Accuracy, 각 공격별 ASR 등)는 `results/` 폴더 내에 CSV 파일(`attack_{dataset}_eps_{eps}.csv`)로 자동 저장되어 실험 결과를 편하게 분석할 수 있습니다.
- 시각화 결과는 마지막 에포크에 `*_vis.png` 파일로 저장됩니다 (예: `fgsm_untargeted_vis.png`).

## ⚙️ Advanced Arguments

`test.py` 스크립트는 정확한 실험 설정을 위해 다음과 같은 주요 인자들도 제공합니다:
- `--eval_interval`: 공격 성능 평가를 수행할 에포크 주기를 설정합니다. (기본값: 3)
- `--seed`: 실험 재현성을 위한 Random Seed (기본값: 42)
- `--pgd_steps`: PGD 공격의 모델 업데이트 반복 횟수 (기본값: 30)
