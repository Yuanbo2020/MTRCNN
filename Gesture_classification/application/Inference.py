import sys, os, argparse
import warnings

warnings.filterwarnings('ignore')

gpu_id = 1
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
from framework.data_generator import *
from framework.processing import *
from framework.models_pytorch import *


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, required=True)
    args = parser.parse_args()

    model_type = args.model

    models = ['CNN_Transformer', 'YAMNet', 'MobileNetV2', 'MTRCNN']
    model_index = models.index(model_type)
    using_models = [CNN_Transformer, YAMNet, MobileNetV2, MTRCNN]

    Dataset_dir = os.path.join(os.getcwd(), 'Dataset')

    model_name = model_type + '.pth'

    model = using_models[model_index](len(config.Gesture_list_name))

    if config.cuda and torch.cuda.is_available():
        model.cuda()

    clip_length = 6
    batch_size = 32
    generator = DataGenerator_gesture(Dataset_dir=Dataset_dir,
                                      renormal=True, clip_length=clip_length, batch_size=batch_size,
                                      test_size=0.115, val_size=0.1)

    print('\nusing model: ', model_type)

    params_num = count_parameters(model)

    event_model_path = os.path.join(os.getcwd(), 'Pretrained_models', model_name)
    model_event = torch.load(event_model_path, map_location='cpu')
    model.load_state_dict(model_event)

    # Generate function
    generate_func = generator.generate_testing(data_type='testing')
    dict = forward(model=model, generate_func=generate_func, cuda=1)

    # softmax classification acc
    average_acc = cal_softmax_classification_accuracy(dict['label'], dict['output'], average='macro')

    print('Params: ' + '%.2f'%(params_num / 1000 ** 2) + ' M; Acc: ', '%.2f'%(average_acc*100), ' %')


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)















