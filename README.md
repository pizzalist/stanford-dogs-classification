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

ex)
- **Chihuahua** <br>
![Chihuahua](https://storage.googleapis.com/wandb-production.appspot.com/letgoofthepizza/images/projects/38075316/a64d3cfb.png?Expires=1703864459&GoogleAccessId=gorilla-files-url-signer-man%40wandb-production.iam.gserviceaccount.com&Signature=MEo7qP%2BwJ5RN9GKIIHrhrOSlMkmQEYD0Lg1%2B5gjXZICu2rnOGNBSL%2BwsZShNfqcc3zuJkdb%2F7RaBD5UH4cVHK9ztbJCmyQpo7PYefqKosUutNIBGsXiymbvTlrAv5rK64VotAGfwXg2YPKhov%2BLxH5ODBPIM1fhgc%2FEY8DYbn2fjd9T6utLZRLxEukMlJT3NEojPWNCt41DhEmPvzwgiDJEbW8%2B2l7CVNqHb%2BRu7YFoDax4u1Zx76XcUnWAwWSOasRpWjSkzHgYY4q4emzd4Kq1gbOimft4i0Owz2IbeSpNg7NW79t1jwEzd3Oh2BGTTSGiFXdBYhGv%2BOKn%2FRyQsXQ%3D%3D) <br>
![Chihuahua2](https://storage.googleapis.com/wandb-production.appspot.com/letgoofthepizza/images/projects/38075316/39c44d33.png?Expires=1703864491&GoogleAccessId=gorilla-files-url-signer-man%40wandb-production.iam.gserviceaccount.com&Signature=LnnTeiY1gCpQljpoB6r63ZCs1DpC%2BkazuWoKhNN5dVZsfnYLsVjeuprWbIlcKp7UNXwG8MXLSNcl5REVOXDs%2BcLWOjNsL59zAfTL02vIaeaEP1V1%2B1eqJVOcNKNYZZdSNr%2F8wMEKgJGMdwBE1JAZSbK2HCZJocy3Tk8BaDdwXMu85VJPUK2YolOmWXxFbiK5kgrRZgLn%2FJFAKjZLGkeVnNZ%2B6a4lkTQtjlpL8hWgUPODddywI1OUatWdq2LIG9%2B58X3JSuB1yBsO8BJcxhpi2iFLJOzHLXwjJt66Yf%2BJNvzLn5dGhF3Sc4InU3TsHk40DW42wdXB%2FQK%2F%2FjnHez2quw%3D%3D)
- **Pug** <br>
![Pug](https://storage.googleapis.com/wandb-production.appspot.com/letgoofthepizza/images/projects/38075316/9b4f5079.png?Expires=1703864776&GoogleAccessId=gorilla-files-url-signer-man%40wandb-production.iam.gserviceaccount.com&Signature=STi2Xtp7x1Sua32RGlpKWNScFdJAbc2Re4JGTM%2FZc4XQ62z7U1piAK6IVIFWWHzlnV09YYe6cPZSIQocqFZAx9Zy8jO%2BNv2%2BeX56BuHQNSA3LaOF1o%2FgPWz1r1jwtmJXj%2BXlsiWdFUyTZCBZxCyZkQq0ewJBp5LyxFWdnj4ricidBigXbPdWcQyepTFz0EZaru%2BLaDR4s0lqbaktj1qw9e3n0ACEEH8Bc0GKlhQQf4%2FctQgNYvmO3dL8AyE46L4O3CpiV0ltNAyZ%2FoSSTKsIUpyHHyIElULjnvD8zltJu7E94J32%2FHBx3FSsY1pmVmW%2FBSfJ2dB86f6q%2BqYzU5D92Q%3D%3D) <br>
![Pug2](https://storage.googleapis.com/wandb-production.appspot.com/letgoofthepizza/images/projects/38075316/e30461b3.png?Expires=1703864803&GoogleAccessId=gorilla-files-url-signer-man%40wandb-production.iam.gserviceaccount.com&Signature=f56XI9HLLI%2BBzrX5S3DDEx4lA6rUFCpfZ2RXGJGiVvL0e9DcAz2dFLR0jn9PD4wyRJm%2B6AwcygaVdOEFM3DseobpCEqpnaAbG4I3MvsljDGakjs0e9mv56tuMfD1M3D4bYql7APKkMrX3ya6o9HS5pKpyHoacyoWD80kQY76DVVQrKOvtz%2F1v4DmLAh1FZ2Cg3lSrm0aImZ8LvrnRKIfzltItuwsEqhSXBNljBEws8NoUrQBgvKwYeHQPtSMtCXodkq1vwFZ%2FBgds4WVI0XgQ5bDEUwcTXGKSUCxa2nVxRL6cFe%2FNLNpCOhq%2FpqJWf3Im3p5KFEVMAZVPkrYLevf%2BA%3D%3D)


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
[Stanford Dog Classification Notion](https://pepper-origami-952.notion.site/Stanford-Dog-Classification-6fcd4c626abe4df79b8bc5e7785d8a2d?pvs=4)# stanford-dogs-classification
