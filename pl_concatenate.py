import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import LambdaLR

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics.sklearns import F1
from pytorch_lightning import Callback
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import ModelCheckpoint

import math
from argparse import ArgumentParser
from src.dataset import *
from src.utils import *
from src.eval_metrics import *
from numpy import load
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"

pl.seed_everything(1111)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class MaxOut(pl.LightningModule):
    def __init__(self, input_dim, output_dim, num_units=2):
        super(MaxOut, self).__init__()
        self.fc1_list = nn.ModuleList([nn.Linear(input_dim, output_dim) for i in range(num_units)])

    def forward(self, x): 

        return self.maxout(x, self.fc1_list)

    def maxout(self, x, layer_list):
        max_output = layer_list[0](x)
        for _, layer in enumerate(layer_list, start=1):
            max_output = torch.max(max_output, layer(x))
        return max_output


class MLPGenreClassifierModel(pl.LightningModule):

    def __init__(self, hyp_params, trial=None):

        super(MLPGenreClassifierModel, self).__init__()
        
        self.save_hyperparameters()
        self.hyp_params = hyp_params
        self.trial = trial
        
        if trial is None:
            dropout = hyp_params.mlp_dropout
        else:
            dropout = trial.suggest_uniform("dropout", 0.0, 0.5)

        if hyp_params.text_embedding_size == hyp_params.image_feature_size:
            self.bn1 = nn.BatchNorm1d(hyp_params.hidden_size)
            self.linear1 = MaxOut(hyp_params.hidden_size, hyp_params.hidden_size)
        else:
            self.bn1 = nn.BatchNorm1d(hyp_params.text_embedding_size+hyp_params.image_feature_size)
            self.linear1 = MaxOut(hyp_params.text_embedding_size+hyp_params.image_feature_size, hyp_params.hidden_size)
        self.drop1 = nn.Dropout(p=dropout)
        
        self.bn2 = nn.BatchNorm1d(hyp_params.hidden_size)
        self.linear2 = MaxOut(hyp_params.hidden_size, hyp_params.hidden_size)
        self.drop2 = nn.Dropout(p=dropout)
        
        self.bn3 = nn.BatchNorm1d(hyp_params.hidden_size)
        self.linear3 = nn.Linear(hyp_params.hidden_size, hyp_params.output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, feature_images=None):
        if feature_images is None:
            x = input_ids
        else:
            x = torch.cat((input_ids, feature_images), dim=1)
        x = self.bn1(x)
        x = self.linear1(x)
        x = self.drop1(x)
        x = self.bn2(x)
        x = self.linear2(x)
        x = self.drop2(x)
        x = self.bn3(x)
        x = self.linear3(x)

        return self.sigmoid(x)
    
    def loss_function(self, outputs, targets):
        return self.hyp_params.criterion(outputs, targets)
    
    def configure_optimizers(self):
        if self.trial is None:
            lr = self.hyp_params.lr
            gamma = 0.856
            #gamma = self.hyp_params.gamma
            when = self.hyp_params.mlp_dropout
        else:
            lr = self.trial.suggest_loguniform('learning_rate', 1e-6, 0.1)
            gamma = self.trial.suggest_uniform("sch_gamma", 0.1, 0.9)
            when = self.trial.suggest_int("when", 1, 10)

        optimizer = getattr(optim, self.hyp_params.optim)(self.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=when, gamma=0.1)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        targets = batch["label"]
        images = batch['image']

        outputs = self(
            input_ids=input_ids,
            feature_images=images
        )
        
        loss = self.loss_function(outputs, targets)
        
        return {'loss': loss}
    
    def train_dataloader(self):
        train_data = MMIMDbDataset(self.hyp_params.data_path,
                               self.hyp_params.dataset,
                               'train'
                              )
        
        train_loader = DataLoader(train_data,
                            batch_size=self.hyp_params.batch_size,
                            shuffle=True,
                            num_workers=0)
        
        return train_loader
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        targets = batch["label"]
        images = batch['image']
        
        outputs = self(
            input_ids=input_ids,
            feature_images=images
        )
        
        loss = self.loss_function(outputs, targets)
        outputs = (outputs > 0.5).float()
        #f1_micro = self.metric(outputs, targets)
        _, f1_micro, f1_macro, f1_weighted, f1_samples = metrics(outputs, targets)
        f1_micro = torch.from_numpy(np.array(f1_micro))
        f1_macro = torch.from_numpy(np.array(f1_macro))
        f1_weighted = torch.from_numpy(np.array(f1_weighted))
        f1_samples = torch.from_numpy(np.array(f1_samples))
        
        return {'val_loss': loss,
                'val_f1_micro': f1_micro,
                'val_f1_macro': f1_macro,
                'val_f1_weighted': f1_weighted,
                'val_f1_samples': f1_samples}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_f1_micro = torch.stack([x['val_f1_micro'] for x in outputs]).mean()
        avg_f1_macro = torch.stack([x['val_f1_macro'] for x in outputs]).mean()
        avg_f1_weighted = torch.stack([x['val_f1_weighted'] for x in outputs]).mean()
        avg_f1_samples = torch.stack([x['val_f1_samples'] for x in outputs]).mean()
        
        '''
        tensorboard_logs = {'loss/val': avg_loss,
                            'f1_micro/val': avg_f1_micro,
                            'f1_macro/val': avg_f1_macro,
                            'f1_weighted/val': avg_f1_weighted,
                            'f1_samples/val': avg_f1_samples}
        '''
        
        self.logger.experiment.add_scalar('f1-micro/val', avg_f1_micro, self.current_epoch+1)
        self.logger.experiment.add_scalar('f1-macro/val', avg_f1_macro, self.current_epoch+1)
        self.logger.experiment.add_scalar('f1-weighted/val', avg_f1_weighted, self.current_epoch+1)
        self.logger.experiment.add_scalar('f1-samples/val', avg_f1_samples, self.current_epoch+1)
        
        return {'val_loss': avg_loss, 'val_f1_micro': avg_f1_micro}
    
    def val_dataloader(self):
        val_data = MMIMDbDataset(self.hyp_params.data_path,
                               self.hyp_params.dataset,
                               'dev'
                              )
        
        val_loader = DataLoader(val_data,
                            batch_size=self.hyp_params.batch_size,
                            num_workers=0)
        
        return val_loader
    
    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        targets = batch["label"]
        images = batch['image']
        
        outputs = self(
            input_ids=input_ids,
            feature_images=images
        )
        
        loss = self.loss_function(outputs.cpu(), targets.cpu())
        outputs = (outputs > 0.5).float()
        _, f1_micro, f1_macro, f1_weighted, f1_samples = metrics(outputs, targets)
        f1_micro = torch.from_numpy(np.array(f1_micro))
        f1_macro = torch.from_numpy(np.array(f1_macro))
        f1_weighted = torch.from_numpy(np.array(f1_weighted))
        f1_samples = torch.from_numpy(np.array(f1_samples))
        f1_per_class = torch.from_numpy(report_per_class(outputs, targets))
        
        return {'test_loss': loss,
                'test_f1_micro': f1_micro,
                'test_f1_macro': f1_macro,
                'test_f1_weighted': f1_weighted,
                'test_f1_samples': f1_samples,
                'test_f1_per_class': f1_per_class}
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_f1_micro = torch.stack([x['test_f1_micro'] for x in outputs]).mean()
        avg_f1_macro = torch.stack([x['test_f1_macro'] for x in outputs]).mean()
        avg_f1_weighted = torch.stack([x['test_f1_weighted'] for x in outputs]).mean()
        avg_f1_samples = torch.stack([x['test_f1_samples'] for x in outputs]).mean()
        avg_f1_per_class = torch.stack([x['test_f1_per_class'] for x in outputs]).mean(0)
        
        print("f1-score per class: ", avg_f1_per_class)
        
        return {'val_loss': avg_loss,
                'test_f1_micro': avg_f1_micro,
                'test_f1_macro': avg_f1_macro,
                'test_f1_weighted': avg_f1_weighted,
                'test_f1_samples': avg_f1_samples}
    
    def test_dataloader(self):
        test_data = MMIMDbDataset(self.hyp_params.data_path,
                               self.hyp_params.dataset,
                               'test'
                              )
        
        test_loader = DataLoader(test_data,
                            batch_size=self.hyp_params.batch_size,
                            num_workers=0)
        
        return test_loader
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--cnn_model', type=str, default="vgg16",
                        help='pretrained CNN to use for image feature extraction')
        parser.add_argument('--image_feature_size', type=int, default=4096,
                            help='image feature size extracted from pretrained CNN (default: 4096)')
        parser.add_argument('--text_embedding_size', type=int, default=300,
                            help='text embedding size used for word2vec model (default: 300)')
        parser.add_argument('--hidden_size', type=int, default=512,
                            help='hidden dimension size (default: 512)')
        parser.add_argument('--mlp_dropout', type=float, default=0.0,
                        help='fully connected layers dropout')
        parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                        help='batch size (default: 8)')
        parser.add_argument('--lr', type=float, default=2e-5,
                            help='initial learning rate (default: 2e-5)')
        parser.add_argument('--optim', type=str, default='Adam',
                            help='optimizer to use (default: Adam)')
        parser.add_argument('--when', type=int, default=2,
                            help='when to decay learning rate (default: 2)')
        parser.add_argument('--gamma', type=int, default=0.5,
                            help='Scheduler factor (default: 0.5)')
        return parser


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


def objective(trial):
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join('checkpoints', "trial_{}".format(trial.number)), monitor="val_f1_micro"
    )

    # The default logger in PyTorch Lightning writes to event files to be consumed by
    # TensorBoard. We create a simple logger instead that holds the log in memory so that the
    # final accuracy can be obtained after optimization. When using the default logger, the
    # final accuracy could be stored in an attribute of the `Trainer` instead.
    metrics_callback = MetricsCallback()
    run_name = create_run_name(args)
    logger = TensorBoardLogger(save_dir='runs_pl_temp/', name=run_name)
    
    '''
    trainer = Trainer.from_argparse_args(args,
                                 logger=logger,
                                 checkpoint_callback=checkpoint_callback,
                                 early_stop_callback=PyTorchLightningPruningCallback(trial, monitor="val_f1_micro"),
                                 callbacks=[metrics_callback],
                                )
    '''
    trainer = pl.Trainer(gpus=1,
                         logger=logger,
                         max_epochs=args.max_epochs,
                         gradient_clip_val=args.gradient_clip_val,
                         checkpoint_callback=checkpoint_callback,
                         early_stop_callback=PyTorchLightningPruningCallback(trial, monitor="val_f1_micro"),
                         callbacks=[metrics_callback],
                        )
    
    mlp = MLPGenreClassifierModel(args, trial)
    trainer.fit(mlp)

    return metrics_callback.metrics[-1]["val_f1_micro"].item()

    
if __name__ == '__main__':
    
    parser = ArgumentParser(description='mmimdb model')

    # add PROGRAM level args
    parser.add_argument('--model', type=str, default='GMU',
                    help='name of the model to use (GMU, Concatenate)')
    parser.add_argument('--dataset', type=str, default='mmimdb',
                        help='dataset to use (default: mmimdb)')
    parser.add_argument('--data_path', type=str, default='data',
                        help='path for storing the dataset')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='frequency of result logging (default: 100)')
    parser.add_argument('--chk', type=int, default=3,
                        help='Number of top models to consider for checkpoint (default: 3)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--no_cuda', action='store_true',
                        help='do not use cuda')
    parser.add_argument('--name', type=str, default='model',
                    help='name of the trial (default: "model")')
    parser.add_argument('--search', action='store_true',
                        help='Hyperparameter search')
    parser.add_argument('--test', action='store_true',
                        help='Load pretrained model for test evaluation')
    
    # add MODEL level args
    parser = MLPGenreClassifierModel.add_model_specific_args(parser)
    
    # add TRAINER level args
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--gradient_clip_val', type=float, default=0.8)
    #parser = Trainer.add_argparse_args(parser)
    
    args = parser.parse_args()
    dataset = str.lower(args.dataset.strip())
    
    use_cuda = False

    output_dim_dict = {
        'meme_dataset': 2,
        'mmimdb': 23
    }

    criterion_dict = {
        'meme_dataset': 'CrossEntropyLoss',
        'mmimdb': 'BCEWithLogitsLoss'
    }

    torch.set_default_tensor_type('torch.FloatTensor')
    if torch.cuda.is_available():
        if args.no_cuda:
            print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
        else:
            torch.cuda.manual_seed(args.seed)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            use_cuda = True

    args.use_cuda = use_cuda
    args.dataset = dataset
    args.model = args.model.strip()
    args.output_dim = output_dim_dict.get(dataset)
    #'''
    label_weights = torch.FloatTensor([1.0,1.649177760375881,2.61128332300062,2.7060713138451655,
    3.6737897950283473,3.9090487238979117,5.229050279329609,5.255146600124766,6.826580226904376,
    6.843216896831844,6.9504950495049505,7.249569707401033,8.613496932515337,10.451612903225806,
    10.690355329949238,12.388235294117647,13.287066246056783,14.37542662116041,16.74751491053678,
    19.914893617021278,22.226912928759894,29.97864768683274,41.7029702970297])
    #'''
    args.criterion = getattr(nn, criterion_dict.get(dataset))(pos_weight=label_weights.cuda())
    #args.criterion = getattr(nn, criterion_dict.get(dataset))()
    
    if args.search:
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=100, timeout=None)

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    
    else:
        if args.test:
            num = 92
            model = MLPGenreClassifierModel.load_from_checkpoint(f'pre_trained_models/{args.name}_{num}.ckpt')
            
            trainer = pl.Trainer(gpus=args.gpus)
            
            trainer.test(model)
            
        else:
            checkpoint_callback = ModelCheckpoint(
                filepath=f'pre_trained_models/{args.name}.ckpt',
                save_top_k=args.chk,
                verbose=True,
                monitor='val_f1_micro',
                mode='max',
                prefix=''
            )
            
            run_name = create_run_name(args)
            logger = TensorBoardLogger(save_dir='runs_pl/', name=run_name)
            
            #trainer = pl.Trainer(gpus=2, distributed_backend='dp', logger=logger, max_epochs=hyp_params.num_epochs)
            trainer = pl.Trainer(gpus=args.gpus,
                                 max_epochs=args.max_epochs,
                                 gradient_clip_val=args.gradient_clip_val,
                                 logger=logger,
                                 checkpoint_callback=checkpoint_callback
                                )

            mlp = MLPGenreClassifierModel(args)

            trainer.fit(mlp)

            trainer.test()