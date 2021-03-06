from src import models
from src.utils import *
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import LambdaLR
from torch import nn
import torch
import numpy as np
import time
import sys

from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from src.eval_metrics import *

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"


####################################################################
#
# Construct the model
#
####################################################################

def initiate(hyp_params, train_loader, valid_loader, test_loader=None):
    model = getattr(models, hyp_params.model+'Model')(hyp_params)

    if hyp_params.use_cuda:
        model = model.cuda()

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()

    lambda1 = lambda epoch: 0.95 ** epoch
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    scheduler = StepLR(optimizer, step_size=hyp_params.when, gamma=0.1)
    #scheduler = LambdaLR(optimizer, step_size=hyp_params.when, gamma=0.5)
    

    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler}

    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)

####################################################################
#
# Training and evaluation scripts
#
####################################################################

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']

    scheduler = settings['scheduler']


    def train(model, optimizer, criterion):
        epoch_loss = 0
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        total_loss = 0.0
        losses = []
        results = []
        truths = []
        correct_predictions = 0
        n_examples = hyp_params.n_train
        start_time = time.time()

        for i_batch, data_batch in enumerate(train_loader):
            input_ids = data_batch["input_ids"]
            targets = data_batch["label"]
            images = data_batch['image']

            model.zero_grad()

            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    input_ids = input_ids.cuda()
                    targets = targets.cuda()
                    images = images.cuda()

            if images.size()[0] != input_ids.size()[0]:
                continue

            outputs = model(
                input_ids=input_ids,
                feature_images=images
            )
    
            if hyp_params.dataset == 'meme_dataset':
                _, preds = torch.max(outputs, dim=1)
            else:
                preds = outputs
            loss = criterion(outputs, targets)
            preds_round = (preds > 0.5).float()
            losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()
            
            total_loss += loss.item() * hyp_params.batch_size
            results.append(preds)
            truths.append(targets)

            proc_loss += loss * hyp_params.batch_size
            proc_size += hyp_params.batch_size
            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                train_acc, train_f1_micro, train_f1_macro, train_f1_weighted, train_f1_samples = metrics(preds_round, targets)
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f} | Train Acc {:5.4f} | Train f1-samples {:5.4f}'.
                      format(epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params.log_interval, avg_loss, train_acc, train_f1_samples))
                proc_loss, proc_size = 0, 0
                start_time = time.time()
        
        avg_loss = total_loss / hyp_params.n_train
        results = torch.cat(results)
        truths = torch.cat(truths)
        return results, truths, avg_loss

    def evaluate(model, criterion, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0

        results = []
        truths = []
        correct_predictions = 0
        n_examples = hyp_params.n_valid

        with torch.no_grad():
            for i_batch, data_batch in enumerate(loader):
                input_ids = data_batch["input_ids"]
                targets = data_batch["label"]
                images = data_batch['image']

                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        input_ids = input_ids.cuda()
                        targets = targets.cuda()
                        images = images.cuda()

                if images.size()[0] != input_ids.size()[0]:
                    continue

                outputs = model(
                    input_ids=input_ids,
                    feature_images=images
                )

                if hyp_params.dataset == 'meme_dataset':
                    _, preds = torch.max(outputs, dim=1)
                else:
                    preds = outputs
                
                total_loss += criterion(outputs, targets).item() * hyp_params.batch_size
                #correct_predictions += torch.sum(preds == targets)

                # Collect the results into dictionary
                results.append(preds)
                truths.append(targets)

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return results, truths, avg_loss

    best_valid = 1e8
    run_name = create_run_name(hyp_params)
    writer = SummaryWriter('runs/'+run_name)
    for epoch in range(1, hyp_params.num_epochs+1):
        start = time.time()
        train_results, train_truths, train_loss = train(model, optimizer, criterion)
        results, truths, val_loss = evaluate(model, criterion, test=False)
        #if test_loader is not None:
        #    results, truths, val_loss = evaluate(model, feature_extractor, criterion, test=True)

        end = time.time()
        duration = end-start

        train_acc, train_f1_micro, train_f1_macro, train_f1_weighted, train_f1_samples = metrics(train_results, train_truths)
        val_acc, val_f1_micro, val_f1_macro, val_f1_weighted, val_f1_samples = metrics(results, truths)
        
        #scheduler.step(val_loss)
        scheduler.step()
        
        print("-"*50)
        print('Epoch {:2d} | Time {:5.4f} sec | Train Loss {:5.4f} | Valid Loss {:5.4f} | Valid Acc {:5.4f} | Valid f1-samples {:5.4f}'.format(epoch, duration, train_loss, val_loss, val_acc, val_f1_samples))
        print("-"*50)
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('F1-micro/train', train_f1_micro, epoch)
        writer.add_scalar('F1-macro/train', train_f1_macro, epoch)
        writer.add_scalar('F1-weighted/train', train_f1_weighted, epoch)
        writer.add_scalar('F1-samples/train', train_f1_samples, epoch)

        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('F1-micro/val', val_f1_micro, epoch)
        writer.add_scalar('F1-macro/val', val_f1_macro, epoch)
        writer.add_scalar('F1-weighted/val', val_f1_weighted, epoch)
        writer.add_scalar('F1-samples/val', val_f1_samples, epoch)

        if val_loss < best_valid:
            print(f"Saved model at pre_trained_models/{hyp_params.name}.pt!")
            #save_model(model, name=hyp_params.name)
            torch.save(model.state_dict(), f'pre_trained_models/{hyp_params.name}.pt')
            best_valid = val_loss

    if test_loader is not None:
        #model = load_model(name=hyp_params.name)
        model = getattr(models, hyp_params.model+'Model')(hyp_params)
        model.load_state_dict(torch.load(f'pre_trained_models/{hyp_params.name}.pt'))
        model.eval()
        results, truths, val_loss = evaluate(model, criterion, test=True)
        test_acc, test_f1_micro, test_f1_macro, test_f1_weighted, test_f1_samples = metrics(results, truths)
        
        print("\n\nTest Acc {:5.4f} | Test f1-micro {:5.4f} | Test f1-macro {:5.4f} | Test f1-weighted {:5.4f} | Test f1-samples {:5.4f}".format(test_acc, test_f1_micro, test_f1_macro, test_f1_weighted, test_f1_samples))

    sys.stdout.flush()
    