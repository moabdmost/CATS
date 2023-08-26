"""Training code for the detector model"""

import argparse
import os
import subprocess
import sys
from itertools import count
from multiprocessing import Process

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from tqdm import tqdm
from transformers import *

from dataset import Corpus, EncodedDataset
# from .download import download
from utils import summary, distributed


def setup_distributed(port=29500):
    if not dist.is_available() or not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        return 0, 1

    if 'MPIR_CVAR_CH3_INTERFACE_HOSTNAME' in os.environ:
        from mpi4py import MPI
        mpi_rank = MPI.COMM_WORLD.Get_rank()
        mpi_size = MPI.COMM_WORLD.Get_size()

        os.environ["MASTER_ADDR"] = '127.0.0.1'
        os.environ["MASTER_PORT"] = str(port)

        dist.init_process_group(backend="nccl", world_size=mpi_size, rank=mpi_rank)
        return mpi_rank, mpi_size

    dist.init_process_group(backend="nccl", init_method="env://")
    return dist.get_rank(), dist.get_world_size()


def load_datasets(data_dir, real_dataset, fake_dataset, tokenizer, batch_size,
                  max_sequence_length, random_sequence_length, epoch_size=None, token_dropout=None, seed=None):

    real_corpus = Corpus(real_dataset, data_dir=data_dir)
    fake_corpus = Corpus(fake_dataset, data_dir=data_dir)

    real_train, real_valid = real_corpus.train, real_corpus.valid
    fake_train, fake_valid = fake_corpus.train, fake_corpus.valid

    Sampler = DistributedSampler if distributed() and dist.get_world_size() > 1 else RandomSampler

    min_sequence_length = 10 if random_sequence_length else None
    train_dataset = EncodedDataset(real_train, fake_train, tokenizer, max_sequence_length, min_sequence_length,
                                   epoch_size, token_dropout, seed)
    train_loader = DataLoader(train_dataset, batch_size, sampler=Sampler(train_dataset), num_workers=0)

    validation_dataset = EncodedDataset(real_valid, fake_valid, tokenizer)
    validation_loader = DataLoader(validation_dataset, batch_size=1, sampler=Sampler(validation_dataset))

    return train_loader, validation_loader



def accuracy_sum(logits, labels):
    if list(logits.shape) == list(labels.shape) + [2]:
        # 2-d outputs
        classification = (logits[..., 0] < logits[..., 1]).long().flatten()
    else:
        classification = (logits > 0).long().flatten()
    assert classification.shape == labels.shape
    return (classification == labels).float().sum().item()


def train(model: nn.Module, optimizer, device: str, loader: DataLoader, desc='Train'):
    model.train()

    train_accuracy = 0
    train_epoch_size = 0
    train_loss = 0

    with tqdm(loader, desc=desc, disable=distributed() and dist.get_rank() > 0) as loop:
        for texts, masks, labels in loop:

            texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
            batch_size = texts.shape[0]

            optimizer.zero_grad()
            output = model(texts, attention_mask=masks, labels=labels)
            loss = output.loss
            logits = output.logits
    
            loss.backward()
            optimizer.step()

            batch_accuracy = accuracy_sum(logits, labels)
            train_accuracy += batch_accuracy
            train_epoch_size += batch_size
            train_loss += loss.item() * batch_size

            loop.set_postfix(loss=loss.item(), acc=train_accuracy / train_epoch_size)

    return {
        "train/accuracy": train_accuracy,
        "train/epoch_size": train_epoch_size,
        "train/loss": train_loss
    }


def validate(model: nn.Module, device: str, loader: DataLoader, votes=1, desc='Validation'):
    model.eval()

    validation_accuracy = 0
    validation_epoch_size = 0
    validation_loss = 0

    records = [record for v in range(votes) for record in tqdm(loader, desc=f'Preloading data ... {v}',
                                                              disable=False)]# disable=dist.is_available() and dist.get_rank() > 0)]
    records = [[records[v * len(loader) + i] for v in range(votes)] for i in range(len(loader))]

    with tqdm(records, desc=desc, disable=distributed() and dist.get_rank() > 0) as loop, torch.no_grad():
        for example in loop:
            losses = []
            logit_votes = []

            for texts, masks, labels in example:
                texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
                batch_size = texts.shape[0]

                loss, logits = model(texts, attention_mask=masks, labels=labels).to_tuple()
                losses.append(loss)
                logit_votes.append(logits)

            loss = torch.stack(losses).mean(dim=0)
            logits = torch.stack(logit_votes).mean(dim=0)

            batch_accuracy = accuracy_sum(logits, labels)
            validation_accuracy += batch_accuracy
            validation_epoch_size += batch_size
            validation_loss += loss.item() * batch_size

            loop.set_postfix(loss=loss.item(), acc=validation_accuracy / validation_epoch_size)

    return {
        "validation/accuracy": validation_accuracy,
        "validation/epoch_size": validation_epoch_size,
        "validation/loss": validation_loss
    }

    
def _all_reduce_dict(tensor_dict, device):
    if distributed():
        for key in tensor_dict.keys():
            if isinstance(tensor_dict[key], torch.Tensor):
                torch.distributed.all_reduce(tensor_dict[key])
            else:
                # Convert non-tensor values to tensor
                tensor_dict[key] = torch.tensor(tensor_dict[key], device=device)

    return tensor_dict

    # wrap in tensor and use reduce to gpu0 tensor
    #output_d = {}
    #for (key, value) in sorted(d.items()):
        #tensor_input = torch.tensor([[value]]).to(device)
        #torch.distributed.all_reduce(tensor_input)
        #output_d[key] = tensor_input.item()
   # return output_d


def run(max_epochs=1,
        device=None,
        batch_size=18, #24
        max_sequence_length=128,
        random_sequence_length=False,
        epoch_size=None,
        seed=None,
        model_keyword = "best-model",
        main_dir = "/workspace/models/",
        data_dir='data',
        real_dataset='real',
        fake_dataset='fake',
        token_dropout=None,
        large=False,
        learning_rate=2e-5,
        weight_decay=0,
        **kwargs):


    args = locals()
    #rank, world_size = setup_distributed()
    rank, world_size = 0, 1  

    if device is None:
        device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'

    print('rank:', rank, 'world_size:', world_size, 'device:', device)

    import torch.distributed as dist
    if distributed() and rank > 0:
        dist.barrier()

    model_name = 'roberta-large' if large else 'roberta-base'
    tokenization_utils.logger.setLevel('ERROR')
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name).to(device)

    if rank == 0:
        summary(model)
        if distributed():
            dist.barrier()

    if world_size > 1:
        model = DistributedDataParallel(model, [rank], output_device=rank, find_unused_parameters=True)

    train_loader, validation_loader = load_datasets(data_dir, real_dataset, fake_dataset, tokenizer, batch_size,
                                                    max_sequence_length, random_sequence_length, epoch_size,
                                                    token_dropout, seed)

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    epoch_loop = count(1) if max_epochs is None else range(1, max_epochs + 1)

    # logdir = os.environ.get("OPENAI_LOGDIR", "logs")
    logdir = main_dir + "models"
    identifier = model_keyword + "_detector"
    os.makedirs(logdir, exist_ok=True)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(logdir, filename_suffix = identifier) if rank == 0 else None
    best_validation_accuracy = 0

    for epoch in epoch_loop:
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)
            validation_loader.sampler.set_epoch(epoch)

        train_metrics = train(model, optimizer, device, train_loader, f'Epoch {epoch}')
        validation_metrics = validate(model, device, validation_loader)

        combined_metrics = _all_reduce_dict({**validation_metrics, **train_metrics}, device)

        combined_metrics["train/accuracy"] /= combined_metrics["train/epoch_size"]
        combined_metrics["train/loss"] /= combined_metrics["train/epoch_size"]
        combined_metrics["validation/accuracy"] /= combined_metrics["validation/epoch_size"]
        combined_metrics["validation/loss"] /= combined_metrics["validation/epoch_size"]

        if rank == 0:
            for key, value in combined_metrics.items():
                writer.add_scalar(key, value, global_step=epoch)

            if combined_metrics["validation/accuracy"] > best_validation_accuracy:
                best_validation_accuracy = combined_metrics["validation/accuracy"]

                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(dict(
                        epoch=epoch,
                        model_state_dict=model_to_save.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(),
                        args=args
                    ),
                    os.path.join(logdir, identifier + ".pt")
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--max-epochs', type=int, default=1)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=18)
    parser.add_argument('--max-sequence-length', type=int, default=128)
    parser.add_argument('--random-sequence-length', action='store_true')
    parser.add_argument('--epoch-size', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--data-dir', type=str, default='/workspace/data')
    parser.add_argument('--main-dir', type=str, default='/workspace/')
    parser.add_argument('--model-keyword', type=str, default="best-model")
    parser.add_argument('--real-dataset', type=str, default='real')
    parser.add_argument('--fake-dataset', type=str, default='fake')
    parser.add_argument('--token-dropout', type=float, default=None)

    parser.add_argument('--large', action='store_true', help='use the roberta-large model instead of roberta-base')
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--weight-decay', type=float, default=0)
    args = parser.parse_args()

    nproc = int(subprocess.check_output([sys.executable, '-c', "import torch;"
                                         "print(torch.cuda.device_count() if torch.cuda.is_available() else 1)"]))
    if nproc > 1:
        print(f'Launching {nproc} processes ...', file=sys.stderr)

        os.environ["MASTER_ADDR"] = '127.0.0.1'
        os.environ["MASTER_PORT"] = str(29500)
        os.environ['WORLD_SIZE'] = str(nproc)
        os.environ['OMP_NUM_THREAD'] = str(1)
        subprocesses = []

        for i in range(nproc):
            os.environ['RANK'] = str(i)
            os.environ['LOCAL_RANK'] = str(i)
            process = Process(target=run, kwargs=vars(args))
            process.start()
            subprocesses.append(process)

        for process in subprocesses:
            process.join()
    else:
        run(**vars(args))
