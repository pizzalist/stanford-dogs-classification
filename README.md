# Stanford Dog Classification Project
[View the Stanford Dogs Dataset Report on Weights & Biases](https://wandb.ai/letgoofthepizza/Stanford%20Dogs%20Dataset/reports/Stanford-dogs-breed---Vmlldzo1MzA2MzQw)
## Introduction
- stanford dogs breed data로 resnet-50의 딥러닝 이미지 분류 학습 성능향상 시키기 
- Task: 개 이미지를 보고 종 분류하기

## Data
### stanford dogs breed 
- Number of categories: 120
- Number of images: 20,580
- 종 이름이 적힌 폴더 안에 해당 사진 있음

ex) <br>
- **Chihuahua** 

<img src=./sample_images/a64d3cfb.png> <br>
<img src=./sample_images/39c44d33.png> <br>

- **Pug** 

<img src=./sample_images/9b4f5079.png> <br>
<img src=./sample_images/e30461b3.png> <br>


## Project Report
본 프로젝트는 wandb로 딥러닝 학습 성능을 확인하였으며 report 또한 다음 wandb report를 활용하였다.
### 목차
1. 학습 데이터의 정규화(Data Normalization)
2. 데이터 증강(Data Augmentation)
3. 가중치 감소(Weight Decay)
4. 학습률 스케쥴러(Learning Rate Scheduler) <br>
5. step 최적화 
6. Learning rate & batch size best 조합 찾기<br>
7. weight decay 최적화
8. pretrained model

[View the Stanford Dogs Dataset Report on Weights & Biases](https://wandb.ai/letgoofthepizza/Stanford%20Dogs%20Dataset/reports/Stanford-dogs-breed---Vmlldzo1MzA2MzQw)

## 세미나 영상 
[Stanford Dog Classification Notion](https://pepper-origami-952.notion.site/Stanford-Dog-Classification-6fcd4c626abe4df79b8bc5e7785d8a2d?pvs=4)
