# Citation
Please consider citing our paper as
```bibtex
@INPROCEEDINGS{10890031,
  author={Hou, Yuanbo and Ren, Qiaoqiao and Wang, Wenwu and Botteldooren, Dick},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Sound-Based Recognition of Touch Gestures and Emotions for Enhanced Human-Robot Interaction}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={Emotion recognition;Privacy;Neural networks;Human-robot interaction;Humanoid robots;Skin;Real-time systems;Low latency communication;Speech processing;Signal resolution;Affective computing;emotion classification;touch gestures;humanoid robots;multi-temporal resolution CNN},
  doi={10.1109/ICASSP49660.2025.10890031}}
```
# Run models

- 1\) Copy Dataset.7z to the application folder under `Emotion_arousal_valence_classification` and `Emotion_gesture_classification`

- 2\) Unzip the Dataset.7z, Meta_data.7z, and Pretrained_models.7z

- 3\) For the emotion arousal valence classification task and the emotion gesture classification task, enter the application folder of `Emotion_arousal_valence_classification` and `Emotion_gesture_classification`, respectively.

# Training

```python 
python Training.py -model CNN_Transformer
```
```python 
python Training.py -model YAMNet
```
```python 
python Training.py -model MobileNetV2
```
```python 
python Training.py -model PANNs
```
```python 
python Training.py -model MTRCNN
```

# Inference

## 1) For the emotion arousal valence classification task

### 1.1) CNN_Transformer 
```python 
python Inference.py -model CNN_Transformer
----------------------------------------------------------------------------------------
using model:  CNN_Transformer
Number of 100 audio clips in testing
Params: 1.58 M; Acc:  29.33  % 
```

### 1.2) YAMNet 
```python 
python Inference.py -model YAMNet
----------------------------------------------------------------------------------------
using model:  YAMNet
Number of 100 audio clips in testing
Params: 3.21 M; Acc:  29.67  %
```

### 1.3) MobileNetV2 
```python 
python Inference.py -model MobileNetV2
----------------------------------------------------------------------------------------
using model:  MobileNetV2
Number of 100 audio clips in testing
Params: 2.23 M; Acc:  45.33  %
```

### 1.4) PANNs 
```python 
python Inference.py -model PANNs
----------------------------------------------------------------------------------------
using model:  PANNs
Number of 100 audio clips in testing
Params: 79.68 M; Acc:  49.67  %
```

### 1.5) PANNs_PreW 
```python 
python Inference.py -model PANNs_PreW
----------------------------------------------------------------------------------------
using model:  PANNs_PreW
Number of 100 audio clips in testing
Params: 79.68 M; Acc:  55.33  %
```

### 1.6) MTRCNN 
```python 
python Inference.py -model MTRCNN
----------------------------------------------------------------------------------------
using model:  MTRCNN
Number of 100 audio clips in testing
Params: 0.24 M; Acc:  56.33  %
```

## 2) For the gesture classification task

### 2.1) CNN_Transformer 
```python 
python Inference.py -model CNN_Transformer
----------------------------------------------------------------------------------------
using model:  CNN_Transformer
Number of 60 audios in testing
Params: 1.58 M; Acc:  61.67  %
```

### 2.2) YAMNet 
```python 
python Inference.py -model YAMNet
----------------------------------------------------------------------------------------
using model:  YAMNet
Number of 60 audios in testing
Params: 3.21 M; Acc:  63.33  %
```

### 2.3) MobileNetV2 
```python 
python Inference.py -model MobileNetV2
----------------------------------------------------------------------------------------
using model:  MobileNetV2
Number of 60 audios in testing
Params: 2.23 M; Acc:  73.33  %
```

### 2.4) PANNs 
```python 
python Inference.py -model PANNs
----------------------------------------------------------------------------------------
The model is large in size, and the uploaded data and models have exceeded 1 GB. Uploading this model further will exceed the GitHub storage limit. Please follow the training instructions to train this model yourself.
```

### 2.5) PANNs_PreW 
```python 
python Inference.py -model PANNs_PreW
----------------------------------------------------------------------------------------
The model is large in size, and the uploaded data and models have exceeded 1 GB. Uploading this model further will exceed the GitHub storage limit. Please follow the training instructions to train this model yourself.
```

### 2.6) MTRCNN 
```python 
python Inference.py -model MTRCNN
----------------------------------------------------------------------------------------
using model:  MTRCNN
Number of 60 audios in testing
Params: 0.24 M; Acc:  85.00  %
```

