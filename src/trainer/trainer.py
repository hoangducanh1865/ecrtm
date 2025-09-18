import numpy as np
import torch
import os
import scipy.io
import yaml
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
from src.models.ECRTM import ECRTM
from src.utils.TextData import TextData
from src.utils import file_utils


class Trainer:
    def __init__(self, args):
        self.args = args
        # loading model configuration
        self.args = file_utils.update_args(self.args, path=self.args.config)
        self.output_prefix = f'output/{self.args.dataset}/{self.args.model}_K{self.args.num_topic}_{self.args.test_index}th'
        file_utils.make_dir(os.path.dirname(self.output_prefix))

        seperate_line_log = '=' * 70
        print(seperate_line_log)
        print(seperate_line_log)
        print('\n' + yaml.dump(vars(self.args), default_flow_style=False))

        self.dataset_handler = TextData(self.args.dataset, self.args.batch_size)

        self.args.vocab_size = self.dataset_handler.train_data.shape[1]
        self.args.word_embeddings = self.dataset_handler.word_embeddings
        
        self.model = ECRTM(args)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def make_optimizer(self):
        args_dict = {
            'params': self.model.parameters(),
            'lr': self.args.learning_rate,
        }

        optimizer = torch.optim.Adam(**args_dict)
        return optimizer

    def make_lr_scheduler(self, optimizer,):
        lr_scheduler = StepLR(optimizer, step_size=self.args.lr_step_size, gamma=0.5)
        return lr_scheduler

    def train(self, data_loader):
        optimizer = self.make_optimizer()

        if "lr_scheduler" in self.args:
            print("===>Warning: use lr_scheduler")
            lr_scheduler = self.make_lr_scheduler(optimizer)

        data_size = len(data_loader.dataset)

        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            loss_rst_dict = defaultdict(float)

            for batch_data in data_loader:

                rst_dict = self.model(batch_data)
                batch_loss = rst_dict['loss']

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                for key in rst_dict:
                    loss_rst_dict[key] += rst_dict[key] * len(batch_data)

            if 'lr_scheduler' in self.args:
                lr_scheduler.step()

            output_log = f'Epoch: {epoch:03d}'
            for key in loss_rst_dict:
                output_log += f' {key}: {loss_rst_dict[key] / data_size :.3f}'

            print(output_log)

        beta = self.model.get_beta().detach().cpu().numpy()
        return beta

    def test(self, input_data):
        data_size = input_data.shape[0]
        theta = np.zeros((data_size, self.args.num_topic))
        all_idx = torch.split(torch.arange(data_size), self.args.batch_size)

        with torch.no_grad():
            self.model.eval()
            for idx in all_idx:
                batch_input = input_data[idx]
                batch_theta = self.model.get_theta(batch_input)
                theta[idx] = batch_theta.cpu()

        return theta

    def print_topic_words(beta, vocab, num_top_word):
        topic_str_list = list()
        for i, topic_dist in enumerate(beta):
            topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(num_top_word + 1):-1]
            topic_str = ' '.join(topic_words)
            topic_str_list.append(topic_str)
            print('Topic {}: {}'.format(i + 1, topic_str))
        return topic_str_list
    
    def fit(self):
        # train model via Trainer.
        beta = self.train(self.dataset_handler.train_loader)

        # print and save topic words.
        topic_str_list = self.print_topic_words(beta, self.dataset_handler.vocab, num_top_word=self.args.num_top_word)
        file_utils.save_text(topic_str_list, path=f'{self.output_prefix}_T{self.args.num_top_word}')

        # save inferred topic distributions of training set and testing set.
        train_theta = self.test(self.dataset_handler.train_data)
        test_theta = self.test(self.dataset_handler.test_data)

        params_dict = {
            'beta': beta,
            'train_theta': train_theta,
            'test_theta': test_theta,
        }

        scipy.io.savemat(f'{self.output_prefix}_params.mat', params_dict)