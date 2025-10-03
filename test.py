"""
Test on single GPU:
    CUDA_VISIBLE_DEVICES=0 python3 test.py --eval_data data_lmdb_release/evaluation --benchmark_all_eval --Transformation None --FeatureExtraction None --SequenceModeling None --Prediction None --Transformer --sensitive --data_filtering_off  --imgH 224 --imgW 224 --TransformerModel=vitstr_small_patch16_224 --saved_model https://github.com/roatienza/deep-text-recognition-benchmark/releases/download/v0.1.0/vitstr_small_patch16_224_aug.pth 

To convert to quantized model, add the ff to the script above:
    --infer_model=vitstr_small_patch16_quant.pt --quantized

To convert to a standalone jit model, add the ff to the script above:
    --infer_model=vitstr_small_patch16_jit.pt 
"""

import os
import time
import string
import re
import validators

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from nltk.metrics.distance import edit_distance

from utils import CTCLabelConverter, AttnLabelConverter, Averager, TokenLabelConverter, get_args
from dataset import hierarchical_dataset, AlignCollate
from model import Model, JitModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def benchmark_all_eval(model, criterion, converter, opt):
    """ Evaluation with benchmark datasets """
    if opt.fast_acc:
        eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_867', 'IC13_1015', 'IC15_2077', 'SVTP', 'CUTE80']
    else:
        eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_860', 'IC03_867', 'IC13_857',
                          'IC13_1015', 'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80']

    evaluation_batch_size = 1 if opt.calculate_infer_time else opt.batch_size

    list_accuracy = []
    total_forward_time = 0
    total_evaluation_data_number = 0
    total_correct_number = 0
    log = open(f'./result/{opt.exp_name}/log_all_evaluation.txt', 'a')
    dashed_line = '-' * 80
    print(dashed_line)
    log.write(dashed_line + '\n')

    for eval_data_name in eval_data_list:
        eval_data_path = os.path.join(opt.eval_data, eval_data_name)
        AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW,
                                               keep_ratio_with_pad=opt.PAD, opt=opt)
        eval_data, eval_data_log = hierarchical_dataset(root=eval_data_path, opt=opt)
        evaluation_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=evaluation_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=AlignCollate_evaluation, pin_memory=False)

        _, accuracy, norm_ED, _, _, _, infer_time, length_of_data = validation(
            model, criterion, evaluation_loader, converter, opt)

        list_accuracy.append(f'{accuracy:0.3f}')
        total_forward_time += infer_time
        total_evaluation_data_number += len(eval_data)
        total_correct_number += accuracy * length_of_data
        log.write(eval_data_log)
        print(f'Acc {accuracy:0.3f}\t normalized_ED {norm_ED:0.3f}')
        log.write(f'Acc {accuracy:0.3f}\t normalized_ED {norm_ED:0.3f}\n')
        print(dashed_line)
        log.write(dashed_line + '\n')

    averaged_forward_time = total_forward_time / total_evaluation_data_number * 1000
    total_accuracy = total_correct_number / total_evaluation_data_number
    params_num = sum([np.prod(p.size()) for p in model.parameters()])

    evaluation_log = 'accuracy: '
    for name, acc in zip(eval_data_list, list_accuracy):
        evaluation_log += f'{name}: {acc}\t'
    evaluation_log += f'total_accuracy: {total_accuracy:0.3f}\t'
    evaluation_log += f'averaged_infer_time: {averaged_forward_time:0.3f}\t# parameters: {params_num/1e6:0.3f}'
    if opt.flops:
        evaluation_log += get_flops(model, opt, converter)
    print(evaluation_log)
    log.write(evaluation_log + '\n')
    log.close()

def clean_text(text):
    """Remove tokens de final de sequência e caracteres indesejados"""
    if '[s]' in text:
        text = text[:text.find('[s]')]
    text = text.strip()  # remove espaços no início/fim
    return text

def generate_predicted_txt(labels, preds_str, filenames, output_path="predicoes.txt"):
    """
    Gera um arquivo TXT no formato: arquivo,verdadeiro,predito
    Labels são limpos de tokens especiais antes de salvar.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("arquivo,verdadeiro,predito\n")  # cabeçalho
        for gt, pred, fname in zip(labels, preds_str, filenames):
            gt_clean = clean_text(gt)
            pred_clean = clean_text(pred)
            f.write(f"{fname},{gt_clean},{pred_clean}\n")
    print(f"[INFO] Arquivo de predições salvo em: {output_path}")

def validation(model, criterion, evaluation_loader, converter, opt):
    """ Validation or evaluation """
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()

    all_labels = []
    all_preds = []
    all_filenames = []

    for i, (image_tensors, labels, filenames) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        length_of_data += batch_size
        image = image_tensors.to(device)

        if opt.Transformer:
            target = converter.encode(labels)
        else:
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
            text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)

        start_time = time.time()

        # CTC prediction
        if 'CTC' in opt.Prediction:
            preds = model(image, text_for_pred)
            forward_time = time.time() - start_time
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            if opt.baiduCTC:
                cost = criterion(preds.permute(1, 0, 2), text_for_loss, preds_size, length_for_loss) / batch_size
            else:
                cost = criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)

            _, preds_index = preds.max(2)
            preds_index = preds_index.view(-1)
            preds_str = converter.decode(preds_index.data, preds_size.data)

        # Transformer prediction
        elif opt.Transformer:
            preds = model(image, text=target, seqlen=converter.batch_max_length)
            _, preds_index = preds.topk(1, dim=-1, largest=True, sorted=True)
            preds_index = preds_index.view(-1, converter.batch_max_length)
            forward_time = time.time() - start_time
            cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))
            length_for_pred = torch.IntTensor([converter.batch_max_length - 1] * batch_size).to(device)
            preds_str = converter.decode(preds_index[:, 1:], length_for_pred)

        # Attention prediction
        else:
            preds = model(image, text_for_pred, is_train=False)
            forward_time = time.time() - start_time
            preds = preds[:, :text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:]
            cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

        infer_time += forward_time
        valid_loss_avg.add(cost)

        # Accuracy & confidence
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []

        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            if opt.Transformer:
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]
                pred_max_prob = pred_max_prob[:pred_EOS]
            elif 'Attn' in opt.Prediction:
                gt = gt[:gt.find('[s]')]
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]
                pred_max_prob = pred_max_prob[:pred_EOS]

            if opt.sensitive and opt.data_filtering_off:
                pred = pred.lower()
                gt = gt.lower()
                alnum = '0123456789abcdefghijklmnopqrstuvwxyz'
                pred = re.sub(f'[^{alnum}]', '', pred)
                gt = re.sub(f'[^{alnum}]', '', gt)

            if pred == gt:
                n_correct += 1

            if len(gt) == 0 or len(pred) == 0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)

            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0
            confidence_score_list.append(confidence_score)
        
        all_labels.extend(labels)
        all_preds.extend(preds_str)
        all_filenames.extend(filenames)

    accuracy = n_correct / float(length_of_data) * 100
    norm_ED = norm_ED / float(length_of_data)

    generate_predicted_txt(all_labels, all_preds, all_filenames)

    return valid_loss_avg.val(), accuracy, norm_ED, preds_str, confidence_score_list, labels, infer_time, length_of_data


def get_state_dict(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove module.
        new_state_dict[name] = v
    return new_state_dict


def get_infer_model(model, opt):
    new_state_dict = get_state_dict(model.state_dict())
    model = JitModel(opt)
    model.load_state_dict(new_state_dict)
    model.eval()

    if opt.quantized:
        if opt.static:
            backend = "qnnpack"
            model.qconfig = torch.quantization.get_default_qconfig(backend)
            torch.backends.quantized.engine = backend
            model_quantized = torch.quantization.prepare(model, inplace=False)
            model_quantized = torch.quantization.convert(model_quantized, inplace=False)
        else:
            from torch.quantization import quantize_dynamic
            model_quantized = quantize_dynamic(model=model,
                                               qconfig_spec={torch.nn.Linear}, dtype=torch.qint8, inplace=False)
        model = torch.jit.script(model_quantized)

    model_scripted = torch.jit.script(model)
    model_scripted.save(opt.infer_model)


def test(opt):
    """ Main test function """
    if opt.Transformer:
        converter = TokenLabelConverter(opt)
    elif 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)

    model = torch.nn.DataParallel(model).to(device)

    print('loading pretrained model from %s' % opt.saved_model)
    if validators.url(opt.saved_model):
        model.load_state_dict(torch.hub.load_state_dict_from_url(opt.saved_model, progress=True, map_location=device))
    else:
        model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    opt.exp_name = '_'.join(opt.saved_model.split('/')[1:])

    if opt.infer_model is not None:
        get_infer_model(model, opt)
        return

    os.makedirs(f'./result/{opt.exp_name}', exist_ok=True)
    os.system(f'cp {opt.saved_model} ./result/{opt.exp_name}/')

    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)

    model.eval()
    opt.eval = True
    with torch.no_grad():
        if opt.benchmark_all_eval:
            benchmark_all_eval(model, criterion, converter, opt)
        else:
            log = open(f'./result/{opt.exp_name}/log_evaluation.txt', 'a')
            AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=opt)
            eval_data, eval_data_log = hierarchical_dataset(root=opt.eval_data, opt=opt)
            evaluation_loader = torch.utils.data.DataLoader(
                eval_data, batch_size=opt.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=AlignCollate_evaluation, pin_memory=False)
            _, accuracy, _, _, _, _, _, _ = validation(model, criterion, evaluation_loader, converter, opt)
            log.write(eval_data_log)
            print(f'{accuracy:0.3f}')
            log.write(f'{accuracy:0.3f}\n')
            log.close()


def get_flops(model, opt, converter):
    from thop import profile
    input = torch.randn(1, 1, opt.imgH, opt.imgW).to(device)
    model = model.to(device)
    if opt.Transformer:
        seqlen = converter.batch_max_length
        text_for_pred = torch.LongTensor(1, seqlen).fill_(0).to(device)
        MACs, params = profile(model, inputs=(input, text_for_pred, True, seqlen))
    else:
        text_for_pred = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(device)
        MACs, params = profile(model, inputs=(input, text_for_pred, ))

    flops = 2 * MACs
    return f'Approximate FLOPS: {flops:0.3f}'


def main(opt=None):
    """ Entry point for run.py or direct execution """
    if opt is None:
        opt = get_args(is_train=False)
        if opt.sensitive:
            opt.character = string.printable[:-6]
    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    test(opt)


if __name__ == "__main__":
    main()
