import numpy as np
import h5py, os, pickle, csv
import time
from framework.utilities import calculate_scalar, scale, create_folder
from sklearn.model_selection import train_test_split


class DataGenerator_emotion_arousal_valence(object):
    def __init__(self, Dataset_dir, renormal, clip_length = 2, batch_size=32, seed = 42, test_size = 0.115, val_size = 0.1):

        # data split
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
        # print('Gesture audio clip num: ', len(Gesture_list))
        # print('Emotion audio clip num: ', len(emotion_list))
        assert len(emotion_list) + len(Gesture_list) == len(audio_names)

        Gesture_list = list(set(Gesture_list))
        emotion_list = list(set(emotion_list))

        # print('Gesture_list: ', Gesture_list)
        # print('emotion_list: ', emotion_list)

        self.Gesture_list_name = ['Tickle', 'Poke', 'Rub', 'Pat', 'Tap', 'Hold']
        self.emotion_list_name = ['Happiness', 'Attention', 'Fear', 'Surprise', 'Confusion',
                                  'Sadness', 'Comfort', 'Calmimg', 'Anger', 'Disgust']

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


        self.emotion_training, self.emotion_validtion, self.emotion_test, \
        self.emotion_training_label, self.emotion_validtion_label, self.emotion_test_label = self.stratified_sampling(
            self.emotion_list_name_dict, self.emotion_list_name, test_size, val_size, seed=seed)

        # print(len(self.emotion_training), len(self.emotion_validtion), len(self.emotion_test))


        self.training_ids = self.emotion_training

        self.batch_size = batch_size
        self.random_state = np.random.RandomState(seed)
        self.validate_random_state = np.random.RandomState(0)
        self.test_random_state = np.random.RandomState(0)

        # Load data
        load_time = time.time()

        ########################################################################################################
        self.val_arousal_labels, self.val_valence_labels, self.val_total = self.get_arousal_valence(self.emotion_validtion)
        self.train_arousal_labels, self.train_valence_labels, self.train_total = self.get_arousal_valence(self.emotion_training)
        self.test_arousal_labels, self.test_valence_labels, self.test_total = self.get_arousal_valence(self.emotion_test)
        ###############################################################################################################

        self.emotion_training_feature = self.load_feature(Dataset_dir, self.emotion_training)
        self.emotion_validtion_feature = self.load_feature(Dataset_dir, self.emotion_validtion)
        self.emotion_test_feature = self.load_feature(Dataset_dir, self.emotion_test)

        self.emotion_training_feature = self.emotion_training_feature[:, :int(clip_length*100)]
        self.emotion_validtion_feature = self.emotion_validtion_feature[:, :int(clip_length*100)]
        self.emotion_test_feature = self.emotion_test_feature[:, :int(clip_length*100)]

        # print('self.emotion_training_feature: ', self.emotion_training_feature.shape)
        # print('self.emotion_validtion_feature: ', self.emotion_validtion_feature.shape)
        # print('self.emotion_test_feature: ', self.emotion_test_feature.shape)

        output_dir = os.path.join(os.getcwd(), '0_normalization_files_' + str(clip_length) + 'seconds')
        # print('output_dir', output_dir)
        create_folder(output_dir)
        normalization_emotion_mel_file = os.path.join(output_dir, 'norm_emotion_mel.pickle')

        if renormal or not os.path.exists(normalization_emotion_mel_file):
            print('normalize......')
            norm_pickle = {}
            (self.mean_emotion_mel, self.std_emotion_mel) = calculate_scalar(np.concatenate(self.emotion_training_feature))
            norm_pickle['mean'] = self.mean_emotion_mel
            norm_pickle['std'] = self.std_emotion_mel
            self.save_pickle(norm_pickle, normalization_emotion_mel_file)

        else:
            print('using: ', normalization_emotion_mel_file)
            norm_pickle = self.load_pickle(normalization_emotion_mel_file)
            self.mean_emotion_mel = norm_pickle['mean']
            self.std_emotion_mel = norm_pickle['std']

        # print(self.mean_emotion_mel)
        # print(self.std_emotion_mel)

        # print("norm: ", self.mean_emotion_mel.shape, self.std_emotion_mel.shape)

        print('Loading data time: {:.3f} s'.format(time.time() - load_time))


    def get_arousal_valence(self, emotion_validtion):
        file = os.path.join(os.getcwd(), 'Meta_data', 'emotion_metadata.csv')

        arousal_valence_dict = {}
        with open(file, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for data in list(reader)[1:]:
                part = data[0].split('\\')
                dir_name = part[1]

                audio_part = data[0].split('\\')[-1].split(',')
                # print(audio_part)  # _Anger_Round_1.wav,0,1,1,1,-1,2
                # # emotion_id	pn	round	arousal_class	valence_class	quadrant

                audio_id = audio_part[0]
                arousal_class = int(audio_part[-3])
                valence_class = int(audio_part[-2])
                quadrant = int(audio_part[-1])

                # print(dir_name, audio_id, arousal_class, valence_class, quadrant)
                # # ['_Anger_Round_1.wav', '0', '1', '1', '1', '-1', '2']
                # # 1 _Anger_Round_1.wav 1 -1 2
                arousal_valence_dict[dir_name + audio_id] = [arousal_class, valence_class, quadrant]

        unvalid_names = ['D:\\Yuanbo\\Code\\4_Gesture\\2024_08_15_audio_yuanbo\\2\\wav_clip\\Pat__Round_3.wav', 'D:\\Yuanbo\\Code\\4_Gesture\\2024_08_15_audio_yuanbo\\2\\wav_clip\\Poke__Round_3.wav', 'D:\\Yuanbo\\Code\\4_Gesture\\2024_08_15_audio_yuanbo\\2\\wav_clip\\Tap__Round_3.wav', 'D:\\Yuanbo\\Code\\4_Gesture\\2024_08_15_audio_yuanbo\\2\\wav_clip\\Tickle__Round_3.wav', 'D:\\Yuanbo\\Code\\4_Gesture\\2024_08_15_audio_yuanbo\\23\\wav_clip\\Tap__Round_3.wav', 'D:\\Yuanbo\\Code\\4_Gesture\\2024_08_15_audio_yuanbo\\25\\wav_clip\\Hold__Round_3.wav', 'D:\\Yuanbo\\Code\\4_Gesture\\2024_08_15_audio_yuanbo\\25\\wav_clip\\Pat__Round_3.wav', 'D:\\Yuanbo\\Code\\4_Gesture\\2024_08_15_audio_yuanbo\\25\\wav_clip\\Poke__Round_3.wav', 'D:\\Yuanbo\\Code\\4_Gesture\\2024_08_15_audio_yuanbo\\25\\wav_clip\\Rub__Round_3.wav', 'D:\\Yuanbo\\Code\\4_Gesture\\2024_08_15_audio_yuanbo\\25\\wav_clip\\Tap__Round_2.wav', 'D:\\Yuanbo\\Code\\4_Gesture\\2024_08_15_audio_yuanbo\\25\\wav_clip\\Tap__Round_3.wav', 'D:\\Yuanbo\\Code\\4_Gesture\\2024_08_15_audio_yuanbo\\25\\wav_clip\\Tickle__Round_3.wav']
        bad_name_list = []
        for name in unvalid_names:
            audio_part = name.split('\\')
            audio_name = audio_part[-3] + '_' + audio_part[-1].replace('__Round_', '_Round_')
            bad_name_list.append(audio_name)

        arousal_labels, valence_labels, quadrant_labels = [], [], []
        for name in emotion_validtion:  # 14_35---_Disgust_Round_3.npy
            if name not in bad_name_list:  #
                part = name.split('---')
                dir_name = part[0].split('_')[0]
                full_name = dir_name + part[-1].replace('.npy', '.wav')
                arousal_class, valence_class, quadrant = arousal_valence_dict[full_name]
                # print(arousal_class, valence_class, quadrant)
                arousal_labels.append(arousal_class)
                valence_labels.append(valence_class)
                quadrant_labels.append(quadrant)
        arousal_labels = np.array(arousal_labels)
        valence_labels = np.array(valence_labels)
        quadrant_labels = np.array(quadrant_labels)

        return arousal_labels+ 1, valence_labels+ 1, quadrant_labels

    def load_feature(self, Dataset_dir, data_path):
        training_feature = []
        for each_path in data_path:
            data = np.load(os.path.join(Dataset_dir, each_path))
            # print(each_path, data.shape)  # 15_1---Hold__Round_2.npy (2, 1001, 64)
            training_feature.append(data)
        return np.array(training_feature)


    def stratified_sampling(self, Gesture_list_name_dict, Gesture_list_name, test_size, val_size, seed):
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

            Gesture_test_label.extend([Gesture_list_name.index(key) for each in test_id])
            Gesture_training_label.extend([Gesture_list_name.index(key) for each in train_id])
            Gesture_validtion_label.extend([Gesture_list_name.index(key) for each in val_id])
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
        audios_num = len(self.emotion_training)

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

            batch_x = self.emotion_training_feature[batch_audio_indexes]
            batch_x = self.transform(batch_x, self.mean_emotion_mel, self.std_emotion_mel)
            # batch_y = [self.emotion_training_label[each] for each in batch_audio_indexes]

            batch_y_arousal = self.train_arousal_labels[batch_audio_indexes]
            batch_y_vanlence = self.train_valence_labels[batch_audio_indexes]
            batch_y_total = self.train_total[batch_audio_indexes]

            yield batch_x, batch_y_arousal, batch_y_vanlence, batch_y_total


    def generate_validate(self, data_type, max_iteration=None):
        # load
        # ------------------ validation --------------------------------------------------------------------------------

        audios_num = len(self.emotion_validtion)

        audio_indexes = [i for i in range(audios_num)]

        self.validate_random_state.shuffle(audio_indexes)

        print('Number of {} audio clips in {}'.format(len(audio_indexes), data_type))

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

            batch_x = self.emotion_validtion_feature[batch_audio_indexes]
            batch_x = self.transform(batch_x, self.mean_emotion_mel, self.std_emotion_mel)

            # batch_y = [self.emotion_validtion_label[each] for each in batch_audio_indexes]
            # batch_y = self.emotion_validtion_label[batch_audio_indexes]

            batch_y_arousal = self.val_arousal_labels[batch_audio_indexes]
            batch_y_vanlence = self.val_valence_labels[batch_audio_indexes]
            batch_y_total = self.val_total[batch_audio_indexes]

            # print(batch_y)

            yield batch_x, batch_y_arousal, batch_y_vanlence, batch_y_total


    def generate_testing(self, data_type, max_iteration=None):
        audios_num = len(self.emotion_test)

        audio_indexes = [i for i in range(audios_num)]

        self.validate_random_state.shuffle(audio_indexes)

        print('Number of {} audio clips in {}'.format(len(audio_indexes), data_type))

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

            batch_x = self.emotion_test_feature[batch_audio_indexes]
            batch_x = self.transform(batch_x, self.mean_emotion_mel, self.std_emotion_mel)

            # batch_y = [self.emotion_test_label[each] for each in batch_audio_indexes]
            # batch_y = self.emotion_test_label[batch_audio_indexes]

            batch_y_arousal = self.test_arousal_labels[batch_audio_indexes]
            batch_y_vanlence = self.test_valence_labels[batch_audio_indexes]

            batch_y_total = self.test_total[batch_audio_indexes]

            # print(batch_y)

            yield batch_x, batch_y_arousal, batch_y_vanlence, batch_y_total


    def transform(self, x, mean, std):
        """Transform data.

        Args:
          x: (batch_x, seq_len, freq_bins) | (seq_len, freq_bins)

        Returns:
          Transformed data.
        """

        return scale(x, mean, std)


