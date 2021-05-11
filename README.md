# pstage_02_image_classification

## Getting Started

### Dependencies

- torch==1.6.0
- torchvision==0.7.0

### Install Requirements

- `pip install -r requirements.txt`

## Code 설명

- Week 1주차 daily mission 코드 구현
- 모든 코드는 직접 작성해보고, 부족한 부분은 이후 공개해주신 코드를 참고했습니다
- `EDA.ipynb` : EDA 구현
- `Augmentation.ipynb` : Augmentation 시각화
- `Darknet.ipynb` : Darknet 구현
- `my_submission.ipynb` : Training, inference 구현
  - 이후 baseline으로 리팩토링 했습니다

---

- `my_submission.ipynb` 코드를 올려주신 baseline 코드를 참고하면서 리팩토링 한 프로젝트 코드입니다
- `dataset.py` : dataset과 transform 정의
- `inference.py` : inference 구현
- `loss.py` : loss 구현
- `model.py` : model 구현
- `train.py`: training, validation 과정 구현, 모델 저장
