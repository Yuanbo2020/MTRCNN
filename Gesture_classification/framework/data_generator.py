import numpy as np
import os, pickle
import time
from framework.utilities import calculate_scalar, scale, create_folder
from sklearn.model_selection import train_test_split


class DataGenerator_gesture(object):
    def __init__(self, Dataset_dir, renormal=True, clip_length=0, batch_size=32, test_size=0.115, val_size=0.1, seed=42):

        emotion_list = []
        Gesture_list = []

        audio_names = []
        audio_name_keys = []
        for file in os.listdir(Dataset_dir):
            part = file.split('---')[1].split('_Round')[0]
            audio_names.append(file)

            sub_part = part.split('_')
            if sub_part[0]:
                Gesture_list.append(sub_part[0])
                audio_name_keys.append(sub_part[0])
            if sub_part[1]:
                emotion_list.append(sub_part[1])
                audio_name_keys.append(sub_part[1])

        print('All audio clip num: ', len(audio_names))
        print('Gesture audio clip num: ', len(Gesture_list))
        print('Emotion audio clip num: ', len(emotion_list))
        assert len(emotion_list) + len(Gesture_list) == len(audio_names)

        Gesture_list = list(set(Gesture_list))
        emotion_list = list(set(emotion_list))

        print('Gesture_list: ', Gesture_list)
        print('emotion_list: ', emotion_list)

        self.Gesture_list_name = ['Tickle', 'Poke', 'Rub', 'Pat', 'Tap', 'Hold']
        self.emotion_list_name = ['Happiness', 'Attention', 'Fear', 'Surprise', 'Confusion', 'Sadness', 'Comfort', 'Calmimg',
                       'Anger', 'Disgust']

        self.Gesture_list_name_dict = {}
        self.emotion_list_name_dict = {}

        for each in self.Gesture_list_name:
            self.Gesture_list_name_dict[each] = []
        for each in self.emotion_list_name:
            self.emotion_list_name_dict[each] = []

        for key, name in zip(audio_name_keys, audio_names):
            # print(key, name)
            if key in self.Gesture_list_name:
                self.Gesture_list_name_dict[key].append(name)
            elif key in self.emotion_list_name:
                self.emotion_list_name_dict[key].append(name)
            else:
                raise Exception("Error!")


        ########################### Gesture ###########################################################################

        self.Gesture_training, self.Gesture_validtion, self.Gesture_test, \
        self.Gesture_training_label, self.Gesture_validtion_label, self.Gesture_test_label = self.stratified_sampling(self.Gesture_list_name_dict, test_size, val_size, seed=seed)

        print(len(self.Gesture_training), len(self.Gesture_validtion), len(self.Gesture_test))


        self.training_ids = self.Gesture_training

        self.batch_size = batch_size
        self.random_state = np.random.RandomState(seed)
        self.validate_random_state = np.random.RandomState(0)
        self.test_random_state = np.random.RandomState(0)

        # Load data
        load_time = time.time()

        self.Gesture_training_feature = self.load_feature(Dataset_dir, self.Gesture_training)
        self.Gesture_validtion_feature = self.load_feature(Dataset_dir, self.Gesture_validtion)
        self.Gesture_test_feature = self.load_feature(Dataset_dir, self.Gesture_test)

        #
        self.Gesture_training_feature = self.Gesture_training_feature[:, :int(clip_length * 100)]
        self.Gesture_validtion_feature = self.Gesture_validtion_feature[:, :int(clip_length * 100)]
        self.Gesture_test_feature = self.Gesture_test_feature[:, :int(clip_length * 100)]

        # print('self.Gesture_training_feature: ', self.Gesture_training_feature.shape)


        output_dir = os.path.join(os.getcwd(), '0_normalization_files_' + str(clip_length) + 'seconds')
        ################################################################################
        # print('output_dir', output_dir)
        create_folder(output_dir)
        normalization_gesture_mel_file = os.path.join(output_dir, 'norm_gesture_mel.pickle')

        if renormal or not os.path.exists(normalization_gesture_mel_file):
            norm_pickle = {}
            (self.mean_gesture_mel, self.std_gesture_mel) = calculate_scalar(np.concatenate(self.Gesture_training_feature))
            norm_pickle['mean'] = self.mean_gesture_mel
            norm_pickle['std'] = self.std_gesture_mel
            self.save_pickle(norm_pickle, normalization_gesture_mel_file)

        else:
            print('using: ', normalization_gesture_mel_file)
            norm_pickle = self.load_pickle(normalization_gesture_mel_file)
            self.mean_gesture_mel = norm_pickle['mean']
            self.std_gesture_mel = norm_pickle['std']

        print(self.mean_gesture_mel)
        print(self.std_gesture_mel)

        print("norm: ", self.mean_gesture_mel.shape, self.std_gesture_mel.shape)

        print('Loading data time: {:.3f} s'.format(time.time() - load_time))


    def load_feature(self, Dataset_dir, data_path):
        training_feature = []
        for each_path in data_path:
            data = np.load(os.path.join(Dataset_dir, each_path))
            # print(each_path, data.shape)  # 15_1---Hold__Round_2.npy (2, 1001, 64)
            training_feature.append(data)
        return np.array(training_feature)


    def stratified_sampling(self, Gesture_list_name_dict, test_size, val_size, seed):
        Gesture_training = []
        Gesture_validtion = []
        Gesture_test = []

        Gesture_training_label = []
        Gesture_validtion_label = []
        Gesture_test_label = []
        for key, value in Gesture_list_name_dict.items():
            # print(key, 'clip num: ', len(value))
            all_id = [i for i in range(len(value))]
            train_val_id, test_id = train_test_split(all_id, test_size=test_size, random_state=seed)
            train_id, val_id = train_test_split(train_val_id, test_size=val_size, random_state=seed)

            Gesture_test.extend([value[each] for each in test_id])
            Gesture_training.extend([value[each] for each in train_id])
            Gesture_validtion.extend([value[each] for each in val_id])

            Gesture_test_label.extend([self.Gesture_list_name.index(key) for each in test_id])
            Gesture_training_label.extend([self.Gesture_list_name.index(key) for each in train_id])
            Gesture_validtion_label.extend([self.Gesture_list_name.index(key) for each in val_id])
        return Gesture_training, Gesture_validtion, Gesture_test, \
               np.array(Gesture_training_label), np.array(Gesture_validtion_label), np.array(Gesture_test_label)


    def load_pickle(self, file):
        with open(file, 'rb') as f:
            data = pickle.load(f)
        return data

    def save_pickle(self, data, file):
        with open(file, 'wb') as f:
            pickle.dump(data, f)

    def generate_training(self):
        audios_num = len(self.Gesture_training)

        audio_indexes = [i for i in range(audios_num)]

        self.random_state.shuffle(audio_indexes)

        iteration = 0
        pointer = 0

        while True:
            if pointer >= audios_num:
                pointer = 0
                self.random_state.shuffle(audio_indexes)

            # Get batch indexes
            batch_audio_indexes = audio_indexes[pointer: pointer + self.batch_size]
            pointer += self.batch_size

            iteration += 1

            batch_x = self.Gesture_training_feature[batch_audio_indexes]
            batch_x = self.transform(batch_x, self.mean_gesture_mel, self.std_gesture_mel)
            # batch_y = [self.Gesture_training_label[each] for each in batch_audio_indexes]
            batch_y = self.Gesture_training_label[batch_audio_indexes]

            yield batch_x, batch_y


    def generate_validate(self, data_type, max_iteration=None):

        audios_num = len(self.Gesture_validtion)

        audio_indexes = [i for i in range(audios_num)]

        self.validate_random_state.shuffle(audio_indexes)

        print('Number of {} audios in {}'.format(len(audio_indexes), data_type))

        iteration = 0
        pointer = 0

        while True:
            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= audios_num:
                break

            batch_audio_indexes = audio_indexes[pointer: pointer + self.batch_size]
            pointer += self.batch_size

            iteration += 1

            batch_x = self.Gesture_validtion_feature[batch_audio_indexes]
            batch_x = self.transform(batch_x, self.mean_gesture_mel, self.std_gesture_mel)

            batch_y = self.Gesture_validtion_label[batch_audio_indexes]

            yield batch_x, batch_y


    def generate_testing(self, data_type, max_iteration=None):

        audios_num = len(self.Gesture_test)

        audio_indexes = [i for i in range(audios_num)]

        self.validate_random_state.shuffle(audio_indexes)

        print('Number of {} audios in {}'.format(len(audio_indexes), data_type))

        iteration = 0
        pointer = 0

        while True:
            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= audios_num:
                break

            batch_audio_indexes = audio_indexes[pointer: pointer + self.batch_size]
            pointer += self.batch_size

            iteration += 1

            batch_x = self.Gesture_test_feature[batch_audio_indexes]
            batch_x = self.transform(batch_x, self.mean_gesture_mel, self.std_gesture_mel)

            batch_y = self.Gesture_test_label[batch_audio_indexes]

            yield batch_x, batch_y


    def transform(self, x, mean, std):
        """Transform data.

        Args:
          x: (batch_x, seq_len, freq_bins) | (seq_len, freq_bins)

        Returns:
          Transformed data.
        """

        return scale(x, mean, std)



