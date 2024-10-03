import sys, os, argparse

sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
from framework.data_generator import *
from framework.processing import *
from framework.models_pytorch import *


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, required=True)
    args = parser.parse_args()

    model_type = args.model

    models = ['CNN_Transformer', 'YAMNet', 'MobileNetV2', 'PANNs', 'MTRCNN']
    model_index = models.index(model_type)
    using_models = [CNN_Transformer, YAMNet, MobileNetV2, PANN, MTRCNN]
    using_model = using_models[model_index]

    lr_init = 1e-3
    batch_size = 32
    config.batch_size = batch_size
    epochs = 100

    sys_name = 'sys_'

    basic_name = sys_name + model_type + '_' + str(lr_init).replace('-', '')

    suffix, system_name = define_system_name(basic_name=basic_name, epochs=epochs, batch_size=batch_size)

    system_path = os.path.join(os.getcwd(), system_name)

    models_dir = os.path.join(system_path, 'md') + '_' + suffix

    log_path = models_dir + '_log'
    create_folder(log_path)
    filename = os.path.basename(__file__).split('.py')[0]
    print_log_file = os.path.join(log_path, filename + '_print.log')
    sys.stdout = Logger(print_log_file, sys.stdout)
    console_log_file = os.path.join(log_path, filename + '_console.log')
    sys.stderr = Logger(console_log_file, sys.stderr)

    model = using_model(5)

    if config.cuda and torch.cuda.is_available():
        model.cuda()

    clip_length = 7
    Dataset_dir = os.path.join(os.getcwd(), 'Dataset')
    generator = DataGenerator_emotion_arousal_valence(Dataset_dir=Dataset_dir,
                                                      renormal=True, clip_length=clip_length, batch_size=batch_size,
                                                      test_size=0.115, val_size=0.1)

    training_process_total(generator, model, models_dir, epochs, batch_size, lr_init=lr_init)
    print('Training is done!!!')


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)















