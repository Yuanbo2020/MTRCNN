import torch

####################################################################################################

cuda = 1

training = 1
testing = 1

if cuda:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
else:
    device = torch.device('cpu')

mel_bins = 64
batch_size = 64
epoch = 1000
lr_init = 1e-3


endswith = '.pth'

Gesture_list_name = ['Tickle', 'Poke', 'Rub', 'Pat', 'Tap', 'Hold']
emotion_list_name = ['Happiness', 'Attention', 'Fear', 'Surprise', 'Confusion',
                                  'Sadness', 'Comfort', 'Calmimg', 'Anger', 'Disgust']

each_emotion_class_num = 1


