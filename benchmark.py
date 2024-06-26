import argparse
from typing import Callable, Dict, Tuple
import time
import numpy as np
import torch
from torch import nn
from dataset import create_dataset

from cuLegKan.layer import BetterLayer as LegendreKANLayer
from original.LegKanLayer import KAL_Layer

from tqdm import tqdm


class cuNet(nn.Module):
    def __init__(self, layers: Tuple[int, int, int], device: str):
        super().__init__()
        self.layer1 = LegendreKANLayer(layers[0], layers[1], 9).to(device)
        self.layer2 = LegendreKANLayer(layers[1], layers[1], 9).to(device)
        self.layer3 = LegendreKANLayer(layers[1], layers[1], 9).to(device)
        self.layer4 = LegendreKANLayer(layers[1], layers[1], 9).to(device)
        self.layer5 = LegendreKANLayer(layers[1], layers[2], 9).to(device)

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x
    

class Net(nn.Module):
    def __init__(self, layers: Tuple[int, int, int], device: str):
        super().__init__()
        self.layer1 = KAL_Layer(layers[0], layers[1], 9).to(device)
        self.layer2 = KAL_Layer(layers[1], layers[1], 9).to(device)
        self.layer3 = KAL_Layer(layers[1], layers[1], 9).to(device)
        self.layer4 = KAL_Layer(layers[1], layers[1], 9).to(device)
        self.layer5 = KAL_Layer(layers[1], layers[2], 9).to(device)

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x
    


def benchmark(
        dataset: Dict[str, torch.Tensor],
        device: str,
        bs: int,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        model: nn.Module,
        reps: int
    ) -> Dict[str, float]:
    forward_times = []
    backward_times = []
    forward_mems = []
    backward_mems = []
    for k in tqdm(range(1 + reps)):
        train_id = np.random.choice(dataset['train_input'].shape[0], bs, replace=False)
        tensor_input = dataset['train_input'][train_id]
        tensor_input = tensor_input.to(device)

        tensor_output = dataset['train_label'][train_id]
        tensor_output = tensor_output.to(device)

        if device == 'cpu':
            t0 = time.time()
            pred = model(tensor_input)
            t1 = time.time()
            if k > 0:
                forward_times.append((t1 - t0) * 1000)
            train_loss = loss_fn(pred, tensor_output)
            t2 = time.time()
            train_loss.backward()
            t3 = time.time()
            if k > 0:
                backward_times.append((t3 - t2) * 1000)
        elif device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            pred = model(tensor_input)
            end.record()

            torch.cuda.synchronize()
            if k > 0:
                forward_times.append(start.elapsed_time(end))
                forward_mems.append(torch.cuda.max_memory_allocated())

            train_loss = loss_fn(pred, tensor_output)

            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            train_loss.backward()
            end.record()

            torch.cuda.synchronize()
            if k > 0:
                backward_times.append(start.elapsed_time(end))
                backward_mems.append(torch.cuda.max_memory_allocated())
    return {
        'forward': np.mean(forward_times),
        'backward': np.mean(backward_times),
        'forward-memory': np.mean(forward_mems) / (1024 ** 3),
        'backward-memory': np.mean(backward_mems) / (1024 ** 3),
    }


def save_results(t: Dict[str, Dict[str, float]], out_path: str):
    maxlen = np.max([len(k) for k in t.keys()])
    with open(out_path, 'w') as f:
        print(f"{' '*maxlen}  |  {'forward':>11}  |  {'backward':>11}  |  {'forward':>11}  |  {'backward':>11}  |  {'num params':>11}  |  {'num trainable params':>20}", file=f)
        print(f"{' '*maxlen}  |  {'forward':>11}  |  {'backward':>11}  |  {'forward':>11}  |  {'backward':>11}  |  {'num params':>11}  |  {'num trainable params':>20}")
        print('-'*130, file=f)
        print('-'*130)
        for key in t.keys():
            print(f"{key:<{maxlen}}  |  {t[key]['forward']:8.2f} ms  |  {t[key]['backward']:8.2f} ms  |  {t[key]['forward-memory']:8.2f} GB  |  {t[key]['backward-memory']:8.2f} GB  |  {t[key]['params']:>11}  |  {t[key]['train_params']:>20}", file=f)
            print(f"{key:<{maxlen}}  |  {t[key]['forward']:8.2f} ms  |  {t[key]['backward']:8.2f} ms  |  {t[key]['forward-memory']:8.2f} GB  |  {t[key]['backward-memory']:8.2f} GB  |  {t[key]['params']:>11}  |  {t[key]['train_params']:>20}")


def count_params(model: nn.Module) -> Tuple[int, int]:
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return pytorch_total_params, pytorch_total_params_train


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', default='times.txt', type=str)
    parser.add_argument('--method', choices=['cu', 'orig', 'all'], type=str)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--inp-size', type=int, default=1440, help='The dimension of the input variables.')
    parser.add_argument('--hid-size', type=int, default=1440, help='The dimension of the hidden layer.')
    parser.add_argument('--reps', type=int, default=10, help='Number of times to repeat execution and average.')
    parser.add_argument('--just-cuda', action='store_true', help='Whether to only execute the cuda version.')
    return parser


def main():
    parser = _create_parser()
    args = parser.parse_args()

    f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
    dataset = create_dataset(
        f, 
        n_var=args.inp_size,
        ranges = [-1,1],
        train_num=1000, 
        test_num=1000,
        normalize_input=False,
        normalize_label=False,
        device='cpu',
        seed=0
    )
    loss_fn = lambda x, y: torch.mean((x - y) ** 2)
    
    res = {}
    
    if args.method == 'cu' or args.method == 'all':
        model = cuNet(layers=[args.inp_size, args.hid_size, 1], device='cpu')
        model.to('cuda')
        res['cuda-gpu'] = benchmark(dataset, 'cuda', args.batch_size, loss_fn, model, args.reps)
        res['cuda-gpu']['params'], res['cuda-gpu']['train_params'] = count_params(model)

    if args.method == 'orig' or args.method == 'all':
        model = Net(layers=[args.inp_size, args.hid_size, 1], device='cpu')
        model.to('cuda')
        res['pytorch-gpu'] = benchmark(dataset, 'cuda', args.batch_size, loss_fn, model, args.reps)
        res['pytorch-gpu']['params'], res['pytorch-gpu']['train_params'] = count_params(model)
    
    save_results(res, args.output_path)

if __name__=='__main__':
    main()