### 2021 BIGDAS Conference

#  📹 An Accurate Extraction of Facial Meta-Information Using Selective Super Resolution from Crowd Images
------------
## Facial Meta-Information Extraction Scheme from Crowd Images

👉 [Paper Download](https://github.com/krapeun/2021-BIGDAS/blob/main/Facial%20Meta-Information%20Extraction%20Scheme%20from%20Crowd%20Images-%20YOO-SUNG%20KIM.pdf)

### 🔎 Why?
  * For crowd monitoring using intelligence video surveillance systems
  * To detect crowd abnormal situations using facial meta-information
<br/><br/>
<br/><br/>

## **⚒ Tech Stacks**
<img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white"/></a>
<img src="https://img.shields.io/badge/Tensorflow-FF6F00?style=flat&logo=Tensorflow&logoColor=white"/></a>
<br/><br/>
<br/><br/>

## **📋 Design**
![image](https://user-images.githubusercontent.com/62520045/171556664-25e46473-2f27-4c4d-875c-e727513c90f8.png)
 1.  Face Detection by [YOLO5Face](https://arxiv.org/abs/2105.12931)
 2.  If lower than threshold? (Low Resolution, Not easy)
 3.  Super Resolution by [ESRGAN](https://arxiv.org/abs/1809.00219) to face images
 4.  Gender Classification by [CNN](https://arxiv.org/abs/1710.07557)
<br/><br/>
<br/><br/>

## **🗂 Dataset**
  * Crowd Dataset in real world
  * Extract total of 551 images by sampling 111 videos from [Crowd-11](https://ieeexplore.ieee.org/document/8015005) Dataset
  * Get annotation files by labeling the x and y coordinate values of face region and gender of the face
  * 6,146 male faces and 4,984 female faces in total 11,130 faces
  * 👉 [Dataset Download](https://github.com/krapeun/2021-BIGDAS/tree/main/Dataset)
<br/><br/>

![image](https://user-images.githubusercontent.com/62520045/171559069-3d0d113c-a7ae-440c-920a-b4ade6b38a18.png)
<br/><br/>
| Case | Number of Face Images | Resolution |
|:----------:|:----------:|:----------:|
| Easy Case | 596 | 48x48 ~ |
| Medium Case | 5,102 | 29x29 ~ 48x48 |
| Hard Case | 5,432 | ~ 29x29 |

<br/><br/>

## **📊 Experiment**

 ### 1. Face Detection Result

| Case | Recall | Accuracy |
|:----------:|:----------:|:----------:|
| Easy Case | 87.67% | 84.41% |
| Medium Case | 85.04% | 80.32% |
| Hard Case | 74.79% | 59.99% |
| Total | 74.79% | 71.1% |

 ### 2. Gender Classification Result

| Case | Before ESRGAN | After ESRGAN |
|:----------:|:----------:|:----------:|
| Easy Case | 72.15% | - |
| Medium Case | 68.86% | 69.78% |
| Hard Case | 65.24% | 65.24% |
| Total | 67.27% | 68.20% |

<br/><br/>

## **👭 Contributions**
| Name | Contribution | Contact |
|:----------:|:----------:|:----------:|
| Jieun Park | Super Resolution, Gender Classification, Dataset, Author | jieunpark@inha.edu |
| Yurim Kang | Face Detection, Dataset | 12171745@inha.edu |
| Yoosung Kim | Corresponding Author | yskim@inha.ac.kr |

<br/><br/>

## **📚 References**
 1.  Crowd-11 : A Dataset for Fine Grained Crowd Behaviour Analysis, IEEE Conference on CVPRW (2017)
 2.  Yang, Shuo and Luo, Ping and Loy, Chen change and Tang, Xiaoou, WIDER FACE: A Face Detection Benchmark, IEEE conference on Computer Vision and Pattern Recognition (CVPR) (2016)
 3.  Delong Qi, Weijun tan, Qi Yao, Jingfeng Liu, YOLO5Face: Why Reinventing a Face Detector, arXiv preprint arXiv:2105.12931 (2021)
 4.  Xingtao Wang and others, ESRGAN: Enhanced Super-Resolution Generative Adversarial Network, arXiv preprint arXiv:1809.00219v2 (2018)
 5.  Octavio Arriaga and Matias Valdenegro-Toro and Paul Plöger, Real-time Convolutional Neural Networks for Emotion and Gender Classification. arXiv preprint arXiv:1710.07557
<br/><br/>
<br/><br/>
