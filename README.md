# Reliable AI: Adversarial Attack Robustness

이 저장소는 MNIST와 CIFAR-10 데이터셋을 사용하여 CNN 모델을 학습하고, 다양한 적대적 공격(Adversarial Attacks)에 대한 모델의 강건성(Robustness)을 평가하는 프로젝트입니다.

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

- 터미널에는 각 공격에 대한 성공 여부(`Success Count`)가 출력됩니다.
- 시각화 결과는 `*_vis.png` 파일로 저장됩니다 (예: `fgsm_untargeted_vis.png`).
