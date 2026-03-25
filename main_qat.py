import argparse
import os
import torch
import json
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from Model.qnets import qtinycnn
from DataUtils.load_data import QD_Dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch implementation of QAT based on Quick, Draw! data.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_root', '-root', type=str, default='Dataset',
                        help='root for the dataset directory.')
    parser.add_argument('--image_size', '-size', type=int, default=28,
                        help='the size of the input image.')

    # training
    parser.add_argument('--epochs', '-e', type=int, default=1,
                        help='number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int,
                        default=256, help='batch size.')
    parser.add_argument('--learning_rate', '-lr', type=float,
                        default=0.1, help='the learning rate.')
    parser.add_argument('--momentum', '-mo', type=float,
                        default=0.9, help='momentum.')
    parser.add_argument('--weight_decay', '-wd', type=float,
                        default=5e-4, help='L2 penalty weight decay.')
    parser.add_argument('--lr_decay_step', '-lrs',
                        type=int, nargs='*', default=[12, 20])
    parser.add_argument('--gamma', '-g', type=float, default=0.1,
                        help='lr is multiplied by gamma on step defined above.')
    parser.add_argument('--ngpu', type=int,
                        default=1, help='0 or less for CPU.')
    
    # quantization
    parser.add_argument('--bit_width', '-bw', type=int,
                        default=8, choices=[8, 4, 2], help='quantization precision.')
    parser.add_argument('--per_channel', action='store_true', default=False,
                        help='use per-channel scaling for weights.')
    parser.add_argument('--quant_input', action='store_true', default=False,
                        help='quantize/binarize the input layer.')
    parser.add_argument('--no_narrow_range', action='store_true', default=False,
                        help='disable narrow range (e.g. for 8-bit quantization use full [-128,127] instead of [-127,127]).')
    parser.add_argument('--export_qonnx', action='store_true', default=False,
                        help='export the best model to QONNX at the end of training.')
    parser.add_argument('--export_qcdq', action='store_true', default=False,
                        help='export the best model to QCDQ ONNX at the end of training.')

    # testing
    parser.add_argument('--test_bs', '-tb', type=int,
                        default=64, help='test batch size.')

    # checkpoint
    parser.add_argument('--save_dir', '-s', type=str,
                        default='./Checkpoints', help='directory for saving checkpoints')

    # for log info
    parser.add_argument('--log', type=str, default='./',
                        help='path of the log info.')

    args = parser.parse_args()

    if not os.path.isdir(args.log):
        os.makedirs(args.log)

    log = open(os.path.join(args.log, f'log_{args.bit_width}bit.txt'), 'w')
    state = {k: v for k, v in args._get_kwargs()}
    log.write(json.dumps(state)+'\n')

    # Init save directory
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    print("*"*50)
    print("Loading the data...")
    train_data = QD_Dataset(mtype="train", root=args.data_root)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True)

    test_data = QD_Dataset(mtype="test", root=args.data_root)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.test_bs, shuffle=True)

    num_classes = train_data.get_number_classes()

    print("Train images number: %d" % len(train_data))
    print("Test images number: %d" % len(test_data))

    net = qtinycnn(num_classes, args.bit_width, 
                   per_channel_scaling=args.per_channel, 
                   quantize_input=args.quant_input,
                   narrow_range=not args.no_narrow_range)

    if args.ngpu > 1:
        net = nn.DataParallel(net)

    if args.ngpu > 0:
        net.cuda()

    print(net)

    optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'],
                                momentum=state['momentum'], weight_decay=state['weight_decay'])

    def train():
        net.train()
        loss_avg = 0.0
        correct = 0
        data_loader = tqdm(train_loader, desc=f'Training {args.bit_width}-bit')
        for batch_idx, (data, target) in enumerate(data_loader):
            if args.ngpu > 0:
                data, target = torch.autograd.Variable(data.cuda()), \
                    torch.autograd.Variable(target.cuda())
            else:
                data, target = torch.autograd.Variable(data.cpu()), \
                    torch.autograd.Variable(target.cpu())

            data = data.view(-1, 1, args.image_size, args.image_size)
            data /= 255.0

            # forward
            output = net(data)

            # backward
            optimizer.zero_grad()
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            # accuracy
            pred = output.data.max(1)[1]
            correct += float(pred.eq(target.data).sum())

            # exponential moving average
            loss_avg = loss_avg*0.2+float(loss)*0.8

        state['train_loss'] = loss_avg
        state['train_accuracy'] = correct/len(train_loader.dataset)

    def test():
        net.eval()
        loss_avg = 0.0
        correct = 0
        data_loader = tqdm(test_loader, desc=f'Testing {args.bit_width}-bit')
        for batch_idx, (data, target) in enumerate(data_loader):
            if args.ngpu > 0:
                data, target = torch.autograd.Variable(data.cuda()), \
                    torch.autograd.Variable(target.cuda())
            else:
                data, target = torch.autograd.Variable(data.cpu()), \
                    torch.autograd.Variable(target.cpu())

            data = data.view(-1, 1, args.image_size, args.image_size)
            data /= 255.0

            # forward
            output = net(data)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += float(pred.eq(target.data).sum())

            # test loss average
            loss_avg += float(loss)

        state['test_loss'] = loss_avg/len(test_loader)
        state['test_accuracy'] = correct/len(test_loader.dataset)

    # Main loop
    best_accuracy = 0.0
    for epoch in range(args.epochs):
        print("")
        print("*"*50)
        print("epoch "+str(epoch+1)+" is running...")
        if epoch+1 in args.lr_decay_step:
            state['learning_rate'] *= args.gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = state['learning_rate']

        state['current_epoch'] = epoch+1
        train()
        print("")
        test()
        print("")
        if state['test_accuracy'] > best_accuracy:
            best_accuracy = state['test_accuracy']
            torch.save(net.state_dict(), os.path.join(
                args.save_dir, f'model_{args.bit_width}bit.pytorch'))
        log.write('%s\n' % json.dumps(state))
        log.flush()
        print(state)
        print(f"Best accuracy: {best_accuracy:.4f}")
        print("*"*50)

    log.close()

    if args.export_qonnx:
        print("*"*50)
        print("Exporting model to QONNX...")
        try:
            from brevitas.export import export_qonnx
            
            # Load the best model
            best_model_path = os.path.join(args.save_dir, f'model_{args.bit_width}bit.pytorch')
            net.load_state_dict(torch.load(best_model_path))
            net.eval()
            
            export_path = os.path.join(args.save_dir, f'model_{args.bit_width}bit.onnx')
            dummy_input = torch.randn(1, 1, args.image_size, args.image_size)
            if args.ngpu > 0:
                dummy_input = dummy_input.cuda()
            
            model_to_export = net.module if isinstance(net, nn.DataParallel) else net
            model_to_export = model_to_export.cpu()
            dummy_input = dummy_input.cpu()
            model_to_export.eval()
            
            export_qonnx(model_to_export, dummy_input, export_path=export_path, dynamo=False)
            print(f"Successfully exported to {export_path}")
        except Exception as e:
            print(f"Failed to export QONNX: {e}")
        print("*"*50)

    if args.export_qcdq:
        print("*"*50)
        print("Exporting model to QCDQ ONNX...")
        try:
            from brevitas.export import export_onnx_qcdq
            
            # Load the best model
            best_model_path = os.path.join(args.save_dir, f'model_{args.bit_width}bit.pytorch')
            net.load_state_dict(torch.load(best_model_path))
            net.eval()
            
            export_path = os.path.join(args.save_dir, f'model_{args.bit_width}bit_qcdq.onnx')
            dummy_input = torch.randn(1, 1, args.image_size, args.image_size)
            
            model_to_export = net.module if isinstance(net, nn.DataParallel) else net
            model_to_export = model_to_export.cpu()
            dummy_input = dummy_input.cpu()
            model_to_export.eval()
            
            export_onnx_qcdq(model_to_export, dummy_input, export_path=export_path, dynamo=False, opset_version=22)
            print(f"Successfully exported to {export_path}")
        except Exception as e:
            print(f"Failed to export QCDQ: {e}")
        print("*"*50)
