import time, os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from framework.utilities import create_folder
from framework.models_pytorch import move_data_to_gpu
import framework.config as config



def define_system_name(alpha=None, basic_name='system', att_dim=None, n_heads=None,
                       batch_size=None, epochs=None):
    suffix = ''
    if alpha:
        suffix = suffix.join([str(each) for each in alpha]).replace('.', '')

    sys_name = basic_name
    sys_suffix = '_b' + str(batch_size) + '_e' + str(epochs) \
                 + '_attd' + str(att_dim) + '_h' + str(n_heads) if att_dim is not None and n_heads is not None \
        else '_b' + str(batch_size)  + '_e' + str(epochs)

    sys_suffix = sys_suffix
    system_name = sys_name + sys_suffix if sys_suffix is not None else sys_name

    return suffix, system_name


def forward(model, generate_func, cuda, return_names = False):
    output = []
    label = []

    audio_names = []
    # Evaluate on mini-batch
    for num, data in enumerate(generate_func):
        (batch_x,  batch_y) = data

        batch_x = move_data_to_gpu(batch_x, cuda)

        model.eval()
        with torch.no_grad():
            output_linear = model(batch_x)

            output.append(output_linear.data.cpu().numpy())
            # ------------------------- labels -------------------------------------------------------------------------
            label.append(batch_y)

    dict = {}

    if return_names:
        dict['audio_names'] = np.concatenate(audio_names, axis=0)

    dict['output'] = np.concatenate(output, axis=0)
    # ----------------------------- labels -------------------------------------------------------------------------
    dict['label'] = np.concatenate(label, axis=0)
    return dict




def cal_softmax_classification_accuracy(target, predict, average=None, eps=1e-8):
    """Calculate accuracy.

    Inputs:
      target: integer array, (audios_num,)
      predict: integer array, (audios_num,)

    Outputs:
      accuracy: float
    """
    # print(target)
    # print(predict)
    classes_num = predict.shape[-1]

    predict = np.argmax(predict, axis=-1)  # (audios_num,)
    samples_num = len(target)


    correctness = np.zeros(classes_num)
    total = np.zeros(classes_num)

    for n in range(samples_num):

        total[target[n]] += 1

        if target[n] == predict[n]:
            correctness[target[n]] += 1

    accuracy = correctness / (total + eps)

    if average == 'each_class':
        return accuracy

    elif average == 'macro':
        return np.mean(accuracy)

    else:
        raise Exception('Incorrect average!')



def evaluate(model, generate_func, cuda):
    # Forward
    dict = forward(model=model, generate_func=generate_func, cuda=cuda)

    # softmax classification acc
    acc = cal_softmax_classification_accuracy(dict['label'], dict['output'], average = 'macro')

    return acc



def training_process_gesture(generator, model, models_dir, epochs, batch_size, lr_init=1e-3, cuda=1):
    create_folder(models_dir)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr_init)

    # ------------------------------------------------------------------------------------------------------------------

    sample_num = len(generator.training_ids)
    one_epoch = int(sample_num / batch_size)
    print('one_epoch: ', one_epoch, 'iteration is 1 epoch')
    print('really batch size: ', batch_size)
    check_iter = one_epoch
    print('validating every: ', check_iter, ' iteration')


    training_start_time = time.time()
    for iteration, all_data in enumerate(generator.generate_training()):
        (batch_x, batch_y) = all_data

        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y = move_data_to_gpu(batch_y, cuda)

        train_bgn_time = time.time()
        model.train()
        optimizer.zero_grad()

        gesture_linear = model(batch_x)

        x_softmax = F.log_softmax(gesture_linear, dim=-1)
        loss = F.nll_loss(x_softmax, batch_y)

        loss.backward()
        optimizer.step()

        Epoch = iteration / one_epoch

        print('epoch: ', '%.3f' % (Epoch), 'loss: %.6f' % float(loss))

        # 6122 / 64 = 95.656
        if iteration % check_iter == 0 and iteration > 1:
            train_fin_time = time.time()
            # Generate function
            generate_func = generator.generate_validate(data_type='validate')
            val_acc = evaluate(model=model, generate_func=generate_func, cuda=cuda)

            print('E: ', '%.3f' % (Epoch), 'val_acc: %.3f' % float(val_acc))

            train_time = train_fin_time - train_bgn_time

            validation_end_time = time.time()
            validate_time = validation_end_time - train_fin_time
            print('epoch: {}, train time: {:.3f} s, iteration time: {:.3f} ms, validate time: {:.3f} s, '
                  'inference time : {:.3f} ms'.format('%.2f' % (Epoch), train_time,
                                                      (train_time / sample_num) * 1000, validate_time,
                                                      1000 * validate_time / sample_num))
            #------------------------ validation done ------------------------------------------------------------------


        # Stop learning
        if iteration > (epochs * one_epoch):
            finish_time = time.time() - training_start_time
            print('Model training finish time: {:.3f} s,'.format(finish_time))
            print("All epochs are done.")

            # correct
            save_out_dict = model.state_dict()
            save_out_path = os.path.join(models_dir, 'model' + config.endswith)
            torch.save(save_out_dict, save_out_path)
            print('Final model saved to {}'.format(save_out_path))

            print('Model training finish time: {:.3f} s,'.format(finish_time))

            print('Training is done!!!')

            break


