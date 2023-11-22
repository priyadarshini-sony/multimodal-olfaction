# this is a generalised one for active learning
from __future__ import print_function

import argparse
import logging
import os
import random as rn
import sys
from collections import defaultdict

import random

import hydra
import mlflow
import mlflow.sklearn
from mlflow import log_artifact, log_metric, log_param

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

import data_preprocessing as preprocessing
import utils as utils
from model.conv import Conv1D
from model.mlp import MLP
from model.mpnn import MPNN
from model.transformer import TRANS
from model.multi import MULTI, MPNN_TRANS_LR
from model.ginet import GINet, MOLCLREMBED_MLP, MOLCLREMBED_LR

#os.environ["PYTHONHASHSEED"] = "2022"
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000/'

glo_seed = 2023
rn.seed(glo_seed)
np.random.seed(glo_seed)
torch.manual_seed(glo_seed)
rng = np.random.RandomState(seed=glo_seed)

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def binary_cross_entropy(y_pred, y_true):
    epsilon = 1e-7  # Small constant to avoid division by zero

    # Ensure dimensions match
    assert y_pred.shape == y_true.shape, "Shapes of y_pred and y_true must be the same."

    # Clip y_pred to avoid numerical instability
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)

    # Compute binary cross entropy element-wise
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # Compute mean across batch dimension
    loss = np.mean(loss, axis=0)

    return loss



def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        _explore_recursive(param_name, element)


def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f"{parent_name}.{k}", v)
            else:
                mlflow.log_param(f"{parent_name}.{k}", v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlflow.log_param(f"{parent_name}.{i}", v)
    else:
        mlflow.log_param(f"{parent_name}", element)

def test_molclr(model, dataloader, criterion, epoch=0):
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []
    print("Testing")
    with torch.no_grad():
        for idx, (g, h, edge, numeric, sequence, embedding, batch_data, label) in enumerate(dataloader):
            _, output = model(batch_data)
            loss = criterion(output, label.to("cuda"))
            # print("Loss:\t", loss)
            total_loss += loss.item()
            y_true.append(label)
            y_pred.append(output)

    total_loss /= len(dataloader.dataset)
    y_true = torch.cat(y_true).detach().cpu().numpy()
    y_pred = torch.cat(y_pred).detach().cpu().numpy()

    if epoch % 10 == 0:
        np.save(f'./pred_npy/molclr_y_pred_np_{epoch}.npy', y_pred)
        np.save(f'./pred_npy/molclr_y_true_np_{epoch}.npy', y_true)

    acc_report = utils.get_eval_report(y_true, y_pred)
    acc_report["loss"] = total_loss
    print("Eval Done")
    return acc_report


def train_molclr(train_data, validation_data, model, optimiser, scheduler, cfg, r, res_dict):
    train_feat, train_label = train_data
    test_feat, test_label = validation_data

    if cfg.train.loss == "custom":
        class_weights = utils.class_weights(train_label)
        criterion = utils.custom_loss(class_weights, cfg)

    elif cfg.train.loss == "binary_crossentropy":
        criterion = nn.BCELoss()

    train_dataset = utils.OlfactortyDataset(train_feat, train_label)
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.train.batchsize, shuffle=True, collate_fn=utils.collate_g, num_workers=1
    )
    test_dataset = utils.OlfactortyDataset(test_feat, test_label)

    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, collate_fn=utils.collate_g, num_workers=1
    )

    for e in range(cfg.train.epochs):
        model.train()

        epoch_loss = 0.0
        for idx, (g, h, edge, numeric, sequence, embedding, batch_data, label) in enumerate(train_dataloader):
            optimiser.zero_grad()
            _, y = model(batch_data)

            loss = criterion(y, label.to("cuda"))
            # print("Loss:\t", loss)
            epoch_loss += loss.item()
            loss.backward()
            optimiser.step()

        epoch_loss = epoch_loss / len(train_label)
        print(f"epoch: {e}\tloss: {epoch_loss}")
        tr_acc = test_molclr(model, train_dataloader, criterion)
        res_dict = utils.log_eval_report("train", tr_acc, mlflow, e, res_dict, r)

        test_acc = test_molclr(model, test_dataloader, criterion, epoch=e)
        
        res_dict = utils.log_eval_report("test", test_acc, mlflow, e, res_dict, r)
        if cfg.train.lr_decay:
            scheduler.step(test_acc["f1-macro"])

        for k1, v1 in tr_acc.items():
            if k1 in [
                    "loss",
                    "f1-macro",
                    "f1-micro",
                    "precision-macro",
                    "recall-macro",
                    "auroc-macro",

                    "var-f1-macro",
                    "var-precision-macro",
                    "var-recall-macro",
                    "var-auroc-macro",
                ]:
                mlflow.log_metric('train-' + k1 +'-in-loop', v1, step=e)
            else:
                mlflow.log_metric('train-' + k1+'-in-loop', v1[-1], step=e)

        for k1, v1 in test_acc.items():
            if k1 in [
                    "loss",
                    "f1-macro",
                    "f1-micro",
                    "precision-macro",
                    "recall-macro",
                    "auroc-macro",

                    "var-f1-macro",
                    "var-precision-macro",
                    "var-recall-macro",
                    "var-auroc-macro",
                ]:
                mlflow.log_metric('test-' + k1+'-in-loop', v1, step=e)
            else:
                mlflow.log_metric('test-' + k1+'-in-loop', v1[-1], step=e)

    return res_dict

def test_molclrembed_lr(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []
    y_pred_embed = []
    y_pred_molclr = []

    embedding_list = []
    molclr_hidden_list = []
    print("Testing")
    with torch.no_grad():
        for idx, (g, h, edge, numeric, sequence, embedding, batch_data, label) in enumerate(dataloader):
            output_embed, output_molclr, output, molclr_hidden = model.forward_test(batch_data, embedding.to("cuda"))
            loss = criterion(output, label.to("cuda"))
            # print("Loss:\t", loss)
            total_loss += loss.item()
            y_true.append(label)
            y_pred.append(output)

            y_pred_embed.append(output_embed)
            y_pred_molclr.append(output_molclr)


            embedding_list.append(embedding)
            molclr_hidden_list.append(molclr_hidden)

    total_loss /= len(dataloader.dataset)
    y_true = torch.cat(y_true).detach().cpu().numpy()
    y_pred = torch.cat(y_pred).detach().cpu().numpy()

    embedding_np = torch.cat(embedding_list).detach().cpu().numpy()
    molclr_hidden_np = torch.cat(molclr_hidden_list).detach().cpu().numpy()

    y_pred_embed = torch.cat(y_pred_embed).detach().cpu().numpy()
    y_pred_molclr = torch.cat(y_pred_molclr).detach().cpu().numpy()

    acc_report = utils.get_eval_report(y_true, y_pred)
    acc_report_embed = utils.get_eval_report(y_true, y_pred_embed)
    acc_report_molclr = utils.get_eval_report(y_true, y_pred_molclr)

    acc_report["loss"] = total_loss
    print("Eval Done")
    return acc_report, acc_report_embed, acc_report_molclr, y_true, embedding_np, molclr_hidden_np


def train_molclrembed_lr(train_data, validation_data, model, optimiser, scheduler, cfg, r, res_dict):
    train_feat, train_label = train_data
    test_feat, test_label = validation_data

    if cfg.train.loss == "custom":
        class_weights = utils.class_weights(train_label)
        criterion = utils.custom_loss(class_weights, cfg)

    elif cfg.train.loss == "binary_crossentropy":
        criterion = nn.BCELoss()

    train_dataset = utils.OlfactortyDataset(train_feat, train_label)
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.train.batchsize, shuffle=True, collate_fn=utils.collate_g, num_workers=1
    )
    test_dataset = utils.OlfactortyDataset(test_feat, test_label)
    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg.train.batchsize, shuffle=False, collate_fn=utils.collate_g, num_workers=1
    )

    for e in range(cfg.train.epochs):
        model.train()

        epoch_loss = 0.0
        for idx, (g, h, edge, numeric, sequence, embedding, batch_data, label) in enumerate(train_dataloader):
            optimiser.zero_grad()
            all_indices, y = model(batch_data, embedding.to("cuda"))
            label = label[:, all_indices]

            loss = criterion(y, label.to("cuda"))

            # print("Loss:\t", loss)
            epoch_loss += loss.item()
            loss.backward()
            optimiser.step()

        epoch_loss = epoch_loss / len(train_label)
        print(f"epoch: {e}\tloss: {epoch_loss}")
        tr_acc, tr_acc_embed, tr_acc_molclr, y_true_train, embedding_np_train, molclr_hidden_np_train = test_molclrembed_lr(model, train_dataloader, criterion)
        res_dict = utils.log_eval_report("train", tr_acc, mlflow, e, res_dict, r)

        test_acc, test_acc_embed, test_acc_molclr, y_true_test, embedding_np_test, molclr_hidden_np_test = test_molclrembed_lr(model, test_dataloader, criterion)
        res_dict = utils.log_eval_report("test", test_acc, mlflow, e, res_dict, r)
        if cfg.train.lr_decay:
            scheduler.step(test_acc["f1-macro"])

        for k1, v1 in tr_acc.items():
            if k1 in [
                    "loss",
                    "f1-macro",
                    "f1-micro",
                    "precision-macro",
                    "recall-macro",
                    "auroc-macro",

                    "var-f1-macro",
                    "var-precision-macro",
                    "var-recall-macro",
                    "var-auroc-macro",
                ]:
                mlflow.log_metric('train-' + k1 +'-in-loop', v1, step=e)
            else:
                mlflow.log_metric('train-' + k1+'-in-loop', v1[-1], step=e)

        for k1, v1 in test_acc.items():
            if k1 in [
                    "loss",
                    "f1-macro",
                    "f1-micro",
                    "precision-macro",
                    "recall-macro",
                    "auroc-macro",

                    "var-f1-macro",
                    "var-precision-macro",
                    "var-recall-macro",
                    "var-auroc-macro",
                ]:
                mlflow.log_metric('test-' + k1+'-in-loop', v1, step=e)
            else:
                mlflow.log_metric('test-' + k1+'-in-loop', v1[-1], step=e)
    return res_dict

def test_molclrembed_mlp(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []
    print("Testing")
    with torch.no_grad():
        for idx, (g, h, edge, numeric, sequence, embedding, batch_data, label) in enumerate(dataloader):
            output = model(batch_data, embedding.to("cuda"))
            loss = criterion(output, label.to("cuda"))
            # print("Loss:\t", loss)
            total_loss += loss.item()
            y_true.append(label)
            y_pred.append(output)
    total_loss /= len(dataloader.dataset)
    y_true = torch.cat(y_true).detach().cpu().numpy()
    y_pred = torch.cat(y_pred).detach().cpu().numpy()
    acc_report = utils.get_eval_report(y_true, y_pred)
    acc_report["loss"] = total_loss
    print("Eval Done")
    return acc_report


def train_molclrembed_mlp(train_data, validation_data, model, optimiser, scheduler, cfg, r, res_dict):
    train_feat, train_label = train_data
    test_feat, test_label = validation_data

    if cfg.train.loss == "custom":
        class_weights = utils.class_weights(train_label)
        criterion = utils.custom_loss(class_weights, cfg)

    elif cfg.train.loss == "binary_crossentropy":
        criterion = nn.BCELoss()

    train_dataset = utils.OlfactortyDataset(train_feat, train_label)
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.train.batchsize, shuffle=True, collate_fn=utils.collate_g, num_workers=1
    )
    test_dataset = utils.OlfactortyDataset(test_feat, test_label)
    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg.train.batchsize, shuffle=False, collate_fn=utils.collate_g, num_workers=1
    )

    for e in range(cfg.train.epochs):
        model.train()

        epoch_loss = 0.0
        for idx, (g, h, edge, numeric, sequence, embedding, batch_data, label) in enumerate(train_dataloader):
            optimiser.zero_grad()
            y = model(batch_data, embedding.to("cuda"))
            loss = criterion(y, label.to("cuda"))
            # print("Loss:\t", loss)
            epoch_loss += loss.item()
            loss.backward()
            optimiser.step()

        epoch_loss = epoch_loss / len(train_label)
        print(f"epoch: {e}\tloss: {epoch_loss}")
        tr_acc = test_molclrembed_mlp(model, train_dataloader, criterion)
        res_dict = utils.log_eval_report("train", tr_acc, mlflow, e, res_dict, r)

        test_acc = test_molclrembed_mlp(model, test_dataloader, criterion)
        res_dict = utils.log_eval_report("test", test_acc, mlflow, e, res_dict, r)
        if cfg.train.lr_decay:
            scheduler.step(test_acc["f1-macro"])

        for k1, v1 in tr_acc.items():
            if k1 in [
                    "loss",
                    "f1-macro",
                    "f1-micro",
                    "precision-macro",
                    "recall-macro",
                    "auroc-macro",

                    "var-f1-macro",
                    "var-precision-macro",
                    "var-recall-macro",
                    "var-auroc-macro",
                ]:
                mlflow.log_metric('train-' + k1 +'-in-loop', v1, step=e)
            else:
                mlflow.log_metric('train-' + k1+'-in-loop', v1[-1], step=e)

        for k1, v1 in test_acc.items():
            if k1 in [
                    "loss",
                    "f1-macro",
                    "f1-micro",
                    "precision-macro",
                    "recall-macro",
                    "auroc-macro",

                    "var-f1-macro",
                    "var-precision-macro",
                    "var-recall-macro",
                    "var-auroc-macro",
                ]:
                mlflow.log_metric('test-' + k1+'-in-loop', v1, step=e)
            else:
                mlflow.log_metric('test-' + k1+'-in-loop', v1[-1], step=e)
    return res_dict

def test_embedding(model, dataloader, criterion, epoch=0):
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []
    print("Testing")
    with torch.no_grad():
        for idx, (g, h, edge, numeric, sequence, embedding, batch_data, label) in enumerate(dataloader):
            output = model(embedding.to("cuda"))
            loss = criterion(output, label.to("cuda"))
            # print("Loss:\t", loss)
            total_loss += loss.item()
            y_true.append(label)
            y_pred.append(output)

    total_loss /= len(dataloader.dataset)
    y_true = torch.cat(y_true).detach().cpu().numpy()
    y_pred = torch.cat(y_pred).detach().cpu().numpy()

    if epoch % 10 == 0:
        np.save(f'./pred_npy/embed_y_pred_np_{epoch}.npy', y_pred)
        np.save(f'./pred_npy/embed_y_true_np_{epoch}.npy', y_true)

    acc_report = utils.get_eval_report(y_true, y_pred)
    acc_report["loss"] = total_loss
    print("Eval Done")
    return acc_report


def train_embedding(train_data, validation_data, model, optimiser, scheduler, cfg, r, res_dict):
    train_feat, train_label = train_data
    test_feat, test_label = validation_data

    if cfg.train.loss == "custom":
        class_weights = utils.class_weights(train_label)
        criterion = utils.custom_loss(class_weights, cfg)

    elif cfg.train.loss == "binary_crossentropy":
        criterion = nn.BCELoss()

    train_dataset = utils.OlfactortyDataset(train_feat, train_label)
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.train.batchsize, shuffle=True, collate_fn=utils.collate_g, num_workers=1
    )
    test_dataset = utils.OlfactortyDataset(test_feat, test_label)
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, collate_fn=utils.collate_g, num_workers=1
    )

    for e in range(cfg.train.epochs):
        model.train()

        epoch_loss = 0.0
        for idx, (g, h, edge, numeric, sequence, embedding, batch_data, label) in enumerate(train_dataloader):
            optimiser.zero_grad()
            y = model(embedding.to("cuda"))
            loss = criterion(y, label.to("cuda"))
            # print("Loss:\t", loss)
            epoch_loss += loss.item()
            loss.backward()
            optimiser.step()

        epoch_loss = epoch_loss / len(train_label)
        print(f"epoch: {e}\tloss: {epoch_loss}")
        tr_acc = test_embedding(model, train_dataloader, criterion)
        res_dict = utils.log_eval_report("train", tr_acc, mlflow, e, res_dict, r)

        test_acc = test_embedding(model, test_dataloader, criterion, epoch=e)
        res_dict = utils.log_eval_report("test", test_acc, mlflow, e, res_dict, r)
        if cfg.train.lr_decay:
            scheduler.step(test_acc["f1-macro"])

        for k1, v1 in tr_acc.items():
            if k1 in [
                    "loss",
                    "f1-macro",
                    "f1-micro",
                    "precision-macro",
                    "recall-macro",
                    "auroc-macro",

                    "var-f1-macro",
                    "var-precision-macro",
                    "var-recall-macro",
                    "var-auroc-macro",
                ]:
                mlflow.log_metric('train-' + k1 +'-in-loop', v1, step=e)
            else:
                mlflow.log_metric('train-' + k1+'-in-loop', v1[-1], step=e)

        for k1, v1 in test_acc.items():
            if k1 in [
                    "loss",
                    "f1-macro",
                    "f1-micro",
                    "precision-macro",
                    "recall-macro",
                    "auroc-macro",

                    "var-f1-macro",
                    "var-precision-macro",
                    "var-recall-macro",
                    "var-auroc-macro",
                ]:
                mlflow.log_metric('test-' + k1+'-in-loop', v1, step=e)
            else:
                mlflow.log_metric('test-' + k1+'-in-loop', v1[-1], step=e)
    return res_dict

def test_numeric(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []
    print("Testing")
    with torch.no_grad():
        for idx, (g, h, edge, numeric, sequence, embedding, batch_data, label) in enumerate(dataloader):
            output = model(numeric.to("cuda"))
            loss = criterion(output, label.to("cuda"))
            # print("Loss:\t", loss)
            total_loss += loss.item()
            y_true.append(label)
            y_pred.append(output)
    total_loss /= len(dataloader.dataset)
    y_true = torch.cat(y_true).detach().cpu().numpy()
    y_pred = torch.cat(y_pred).detach().cpu().numpy()
    acc_report = utils.get_eval_report(y_true, y_pred)
    acc_report["loss"] = total_loss
    print("Eval Done")
    return acc_report


def train_numeric(train_data, validation_data, model, optimiser, scheduler, cfg, r, res_dict):

    train_feat, train_label = train_data
    test_feat, test_label = validation_data

    if cfg.train.loss == "custom":
        class_weights = utils.class_weights(train_label)
        criterion = utils.custom_loss(class_weights, cfg)

    elif cfg.train.loss == "binary_crossentropy":
        criterion = nn.BCELoss()

    train_dataset = utils.OlfactortyDataset(train_feat, train_label)
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.train.batchsize, shuffle=True, collate_fn=utils.collate_g, num_workers=1
    )
    test_dataset = utils.OlfactortyDataset(test_feat, test_label)
    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg.train.batchsize, shuffle=False, collate_fn=utils.collate_g, num_workers=1
    )

    for e in range(cfg.train.epochs):
        model.train()

        epoch_loss = 0.0
        for idx, (g, h, edge, numeric, sequence, embedding, batch_data, label) in enumerate(train_dataloader):
            optimiser.zero_grad()
            y = model(numeric.to("cuda"))
            loss = criterion(y, label.to("cuda"))
            # print("Loss:\t", loss)
            epoch_loss += loss.item()
            loss.backward()
            optimiser.step()

        epoch_loss = epoch_loss / len(train_label)
        print(f"epoch: {e}\tloss: {epoch_loss}")
        tr_acc = test_numeric(model, train_dataloader, criterion)
        res_dict = utils.log_eval_report("train", tr_acc, mlflow, e, res_dict, r)

        test_acc = test_numeric(model, test_dataloader, criterion)
        res_dict = utils.log_eval_report("test", test_acc, mlflow, e, res_dict, r)
        if cfg.train.lr_decay:
            scheduler.step(test_acc["f1-macro"])

        for k1, v1 in tr_acc.items():
            if k1 in [
                    "loss",
                    "f1-macro",
                    "f1-micro",
                    "precision-macro",
                    "recall-macro",
                    "auroc-macro",

                    "var-f1-macro",
                    "var-precision-macro",
                    "var-recall-macro",
                    "var-auroc-macro",
                ]:
                mlflow.log_metric('train-' + k1 +'-in-loop', v1, step=e)
            else:
                mlflow.log_metric('train-' + k1+'-in-loop', v1[-1], step=e)

        for k1, v1 in test_acc.items():
            if k1 in [
                    "loss",
                    "f1-macro",
                    "f1-micro",
                    "precision-macro",
                    "recall-macro",
                    "auroc-macro",

                    "var-f1-macro",
                    "var-precision-macro",
                    "var-recall-macro",
                    "var-auroc-macro",
                ]:
                mlflow.log_metric('test-' + k1+'-in-loop', v1, step=e)
            else:
                mlflow.log_metric('test-' + k1+'-in-loop', v1[-1], step=e)
    return res_dict
    
def test_graph(model, dataloader, criterion):

    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []
    print("Testing")
    with torch.no_grad():
        for idx, (g, h, edge, numeric, sequence, embedding, molclr, label) in enumerate(dataloader):
            g, h, edge = g.cuda(), h.cuda(), edge.cuda()
            g, h, edge = (
                Variable(g),
                Variable(h),
                Variable(edge),
            )

            if edge.size(3) != 0:
                output = model(g, h, edge)
                loss = criterion(output, label.to("cuda"))
                # print("Loss:\t", loss)
                total_loss += loss.item()
                y_true.append(label)
                y_pred.append(output)
            else:
                print('-' * 50)
                print('detected zero dimensional vector - skipping')
                print('-' * 50)
    total_loss /= len(dataloader.dataset)
    y_true = torch.cat(y_true).detach().cpu().numpy()
    y_pred = torch.cat(y_pred).detach().cpu().numpy()
    acc_report = utils.get_eval_report(y_true, y_pred)
    acc_report["loss"] = total_loss
    print("Eval Done")
    return acc_report


def train_graph(train_data, validation_data, model, optimiser, scheduler, cfg, r, res_dict):

    train_feat, train_label = train_data
    test_feat, test_label = validation_data

    if cfg.train.loss == "custom":
        class_weights = utils.class_weights(train_label)
        criterion = utils.custom_loss(class_weights, cfg)

    elif cfg.train.loss == "binary_crossentropy":
        criterion = nn.BCELoss()

    train_dataset = utils.OlfactortyDataset(train_feat, train_label)
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.train.batchsize, shuffle=True, collate_fn=utils.collate_g, num_workers=1
    )
    test_dataset = utils.OlfactortyDataset(test_feat, test_label)
    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg.train.batchsize, shuffle=False, collate_fn=utils.collate_g, num_workers=1
    )

    for e in range(cfg.train.epochs):
        model.train()

        epoch_loss = 0.0
        for idx, (g, h, edge, numeric, sequence, embedding, molclr, label) in enumerate(train_dataloader):
            optimiser.zero_grad()
            g, h, edge = g.cuda(), h.cuda(), edge.cuda()
            g, h, edge = (
                Variable(g),
                Variable(h),
                Variable(edge),
            )

            if edge.size(3) != 0:
                y = model(g, h, edge)
                loss = criterion(y, label.to("cuda"))
                # print("Loss:\t", loss)
                epoch_loss += loss.item()
                loss.backward()
                optimiser.step()
            else:
                print('-' * 50)
                print('detected zero dimensional vector - skipping')
                print('-' * 50)

        epoch_loss = epoch_loss / len(train_label)
        print(f"epoch: {e}\tloss: {epoch_loss}")
        tr_acc = test_graph(model, train_dataloader, criterion)
        res_dict = utils.log_eval_report("train", tr_acc, mlflow, e, res_dict, r)

        test_acc = test_graph(model, test_dataloader, criterion)
        res_dict = utils.log_eval_report("test", test_acc, mlflow, e, res_dict, r)
        if cfg.train.lr_decay:
            scheduler.step(test_acc["f1-macro"])

        for k1, v1 in tr_acc.items():
            if k1 in [
                    "loss",
                    "f1-macro",
                    "f1-micro",
                    "precision-macro",
                    "recall-macro",
                    "auroc-macro",

                    "var-f1-macro",
                    "var-precision-macro",
                    "var-recall-macro",
                    "var-auroc-macro",
                ]:
                mlflow.log_metric('train-' + k1 +'-in-loop', v1, step=e)
            else:
                mlflow.log_metric('train-' + k1+'-in-loop', v1[-1], step=e)

        for k1, v1 in test_acc.items():
            if k1 in [
                    "loss",
                    "f1-macro",
                    "f1-micro",
                    "precision-macro",
                    "recall-macro",
                    "auroc-macro",

                    "var-f1-macro",
                    "var-precision-macro",
                    "var-recall-macro",
                    "var-auroc-macro",
                ]:
                mlflow.log_metric('test-' + k1+'-in-loop', v1, step=e)
            else:
                mlflow.log_metric('test-' + k1+'-in-loop', v1[-1], step=e)

    return res_dict
    
def test_sequence(model, dataloader, criterion):

    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []
    print("Testing")
    with torch.no_grad():
        for idx, (g, h, edge, numeric, sequence, embedding, molclr, label) in enumerate(dataloader):
            sequence= sequence.to(torch.float32).cuda()
            
            output = model(sequence)
            loss = criterion(output, label.to(torch.float).to("cuda"))
            total_loss += loss.item()
            y_true.append(label)
            y_pred.append(output)
    total_loss /= len(dataloader.dataset)
    y_true = torch.cat(y_true).detach().cpu().numpy()
    y_pred = torch.cat(y_pred).detach().cpu().numpy()
    acc_report = utils.get_eval_report(y_true, y_pred)
    acc_report["loss"] = total_loss
    print("Eval Done")
    return acc_report


def train_sequence(train_data, validation_data, model, optimiser, scheduler, cfg, r, res_dict):

    train_feat, train_label = train_data
    test_feat, test_label = validation_data

    if cfg.train.loss == "custom":
        class_weights = utils.class_weights(train_label)
        criterion = utils.custom_loss(class_weights, cfg)

    elif cfg.train.loss == "binary_crossentropy":
        criterion = nn.BCELoss()
        
    train_dataset = utils.OlfactortyDataset(train_feat, train_label)
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.train.batchsize, shuffle=True, collate_fn=utils.collate_g, num_workers=1
    )
    test_dataset = utils.OlfactortyDataset(test_feat, test_label)
    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg.train.batchsize, shuffle=False, collate_fn=utils.collate_g, num_workers=1
    )

    for e in range(cfg.train.epochs):
        model.train()
        epoch_loss = 0.0

        for idx, (g, h, edge, numeric, sequence, embedding, molclr, label) in enumerate(train_dataloader):
            optimiser.zero_grad()

            sequence= sequence.to(torch.float32).cuda()
            
            label = label.to(torch.float).to("cuda")
            class_out = model(sequence)
            loss = criterion(class_out, label) 
            # print("Loss:\t", loss)
            epoch_loss += loss.item()
            loss.backward()
            optimiser.step()
            
        epoch_loss = epoch_loss / len(train_label)
        print(f"epoch: {e}\tloss: {epoch_loss}")
        tr_acc = test_sequence(model, train_dataloader, criterion)
        res_dict = utils.log_eval_report("train", tr_acc, mlflow, e, res_dict, r)

        test_acc = test_sequence(model, test_dataloader, criterion)
        res_dict = utils.log_eval_report("test", test_acc, mlflow, e, res_dict, r)
        if cfg.train.lr_decay:
            scheduler.step(test_acc["f1-macro"])

        for k1, v1 in tr_acc.items():
            if k1 in [
                    "loss",
                    "f1-macro",
                    "f1-micro",
                    "precision-macro",
                    "recall-macro",
                    "auroc-macro",

                    "var-f1-macro",
                    "var-precision-macro",
                    "var-recall-macro",
                    "var-auroc-macro",
                ]:
                mlflow.log_metric('train-' + k1 +'-in-loop', v1, step=e)
            else:
                mlflow.log_metric('train-' + k1+'-in-loop', v1[-1], step=e)

        for k1, v1 in test_acc.items():
            if k1 in [
                    "loss",
                    "f1-macro",
                    "f1-micro",
                    "precision-macro",
                    "recall-macro",
                    "auroc-macro",

                    "var-f1-macro",
                    "var-precision-macro",
                    "var-recall-macro",
                    "var-auroc-macro",
                ]:
                mlflow.log_metric('test-' + k1+'-in-loop', v1, step=e)
            else:
                mlflow.log_metric('test-' + k1+'-in-loop', v1[-1], step=e)
    return res_dict 
    
def test_multi(model, dataloader, criterion):

    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []
    print("Testing")
    with torch.no_grad():
        for idx, (g, h, edge, numeric, sequence, embedding, molclr, label) in enumerate(dataloader):
            g, h, edge, numeric, sequence= g.cuda(), h.cuda(), edge.cuda(), numeric.cuda(), sequence.to(torch.float32).cuda()
            g, h, edge, numeric, sequence= (
                Variable(g),
                Variable(h),
                Variable(edge),
                Variable(numeric),
                Variable(sequence),
            )

            if edge.size(3) != 0:
                output = model(g, h, edge, numeric,sequence)
                loss = criterion(output, label.to("cuda"))
                # print("Loss:\t", loss)
                total_loss += loss.item()
                y_true.append(label)
                y_pred.append(output)
            else:
                print('-' * 50)
                print('detected zero dimensional vector - skipping')
                print('-' * 50)

    total_loss /= len(dataloader.dataset)
    y_true = torch.cat(y_true).detach().cpu().numpy()
    y_pred = torch.cat(y_pred).detach().cpu().numpy()
    acc_report = utils.get_eval_report(y_true, y_pred)
    acc_report["loss"] = total_loss
    print("Eval Done")
    return acc_report


def train_multi(train_data, validation_data, model, optimiser, scheduler, cfg, r, res_dict):

    train_feat, train_label = train_data
    test_feat, test_label = validation_data

    if cfg.train.loss == "custom":
        class_weights = utils.class_weights(train_label)
        criterion = utils.custom_loss(class_weights, cfg)

    elif cfg.train.loss == "binary_crossentropy":
        criterion = nn.BCELoss()

    train_dataset = utils.OlfactortyDataset(train_feat, train_label)
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.train.batchsize, shuffle=True, collate_fn=utils.collate_g, num_workers=1
    )
    test_dataset = utils.OlfactortyDataset(test_feat, test_label)
    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg.train.batchsize, shuffle=False, collate_fn=utils.collate_g, num_workers=1
    )

    for e in range(cfg.train.epochs):
        model.train()

        epoch_loss = 0.0
        for idx, (g, h, edge, numeric, sequence, embedding, molclr, label) in enumerate(train_dataloader):
            optimiser.zero_grad()
            g, h, edge, numeric, sequence= g.cuda(), h.cuda(), edge.cuda(), numeric.cuda(), sequence.to(torch.float32).cuda()
            g, h, edge, numeric, sequence= (
                Variable(g),
                Variable(h),
                Variable(edge),
                Variable(numeric),
                Variable(sequence),
            )

            if edge.size(3) != 0:
                y = model(g, h, edge, numeric,sequence)
                loss = criterion(y, label.to("cuda"))
                # print("Loss:\t", loss)
                epoch_loss += loss.item()
                loss.backward()
                optimiser.step()
            else:
                print('-' * 50)
                print('detected zero dimensional vector - skipping')
                print('-' * 50)


        epoch_loss = epoch_loss / len(train_label)
        print(f"epoch: {e}\tloss: {epoch_loss}")
        tr_acc = test_multi(model, train_dataloader, criterion)
        res_dict = utils.log_eval_report("train", tr_acc, mlflow, e, res_dict, r)

        test_acc = test_multi(model, test_dataloader, criterion)
        res_dict = utils.log_eval_report("test", test_acc, mlflow, e, res_dict, r)
        if cfg.train.lr_decay:
            scheduler.step(test_acc["f1-macro"])


        for k1, v1 in tr_acc.items():
            if k1 in [
                    "loss",
                    "f1-macro",
                    "f1-micro",
                    "precision-macro",
                    "recall-macro",
                    "auroc-macro",

                    "var-f1-macro",
                    "var-precision-macro",
                    "var-recall-macro",
                    "var-auroc-macro",
                ]:
                mlflow.log_metric('train-' + k1 +'-in-loop', v1, step=e)
            else:
                mlflow.log_metric('train-' + k1+'-in-loop', v1[-1], step=e)

        for k1, v1 in test_acc.items():
            if k1 in [
                    "loss",
                    "f1-macro",
                    "f1-micro",
                    "precision-macro",
                    "recall-macro",
                    "auroc-macro",

                    "var-f1-macro",
                    "var-precision-macro",
                    "var-recall-macro",
                    "var-auroc-macro",
                ]:
                mlflow.log_metric('test-' + k1+'-in-loop', v1, step=e)
            else:
                mlflow.log_metric('test-' + k1+'-in-loop', v1[-1], step=e)
        
    return res_dict

def test_multi_dropout(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []
    print("Testing")
    with torch.no_grad():
        for idx, (g, h, edge, numeric, sequence, embedding, molclr, label) in enumerate(dataloader):
            g, h, edge, numeric, sequence= g.cuda(), h.cuda(), edge.cuda(), numeric.cuda(), sequence.to(torch.float32).cuda()
            g, h, edge, numeric, sequence= (
                Variable(g),
                Variable(h),
                Variable(edge),
                Variable(numeric),
                Variable(sequence),
            )

            if edge.size(3) != 0:
                output = model.forward_test(g, h, edge, numeric,sequence)
                loss = criterion(output, label.to("cuda"))
                # print("Loss:\t", loss)
                total_loss += loss.item()
                y_true.append(label)
                y_pred.append(output)
            else:
                print('-' * 50)
                print('detected zero dimensional vector - skipping')
                print('-' * 50)

    total_loss /= len(dataloader.dataset)
    y_true = torch.cat(y_true).detach().cpu().numpy()
    y_pred = torch.cat(y_pred).detach().cpu().numpy()
    acc_report = utils.get_eval_report(y_true, y_pred)
    acc_report["loss"] = total_loss
    print("Eval Done")
    return acc_report


def train_multi_lr(train_data, validation_data, model, optimiser, scheduler, cfg, r, res_dict):

    train_feat, train_label = train_data
    test_feat, test_label = validation_data

    if cfg.train.loss == "custom":
        class_weights = utils.class_weights(train_label)
        criterion = utils.custom_loss(class_weights, cfg)

    elif cfg.train.loss == "binary_crossentropy":
        criterion = nn.BCELoss()

    train_dataset = utils.OlfactortyDataset(train_feat, train_label)
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.train.batchsize, shuffle=True, collate_fn=utils.collate_g, num_workers=1
    )
    test_dataset = utils.OlfactortyDataset(test_feat, test_label)
    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg.train.batchsize, shuffle=False, collate_fn=utils.collate_g, num_workers=1
    )

    for e in range(cfg.train.epochs):
        model.train()

        epoch_loss = 0.0
        for idx, (g, h, edge, numeric, sequence, embedding, molclr, label) in enumerate(train_dataloader):
            optimiser.zero_grad()
            g, h, edge, numeric, sequence= g.cuda(), h.cuda(), edge.cuda(), numeric.cuda(), sequence.to(torch.float32).cuda()
            g, h, edge, numeric, sequence= (
                Variable(g),
                Variable(h),
                Variable(edge),
                Variable(numeric),
                Variable(sequence),
            )
            if edge.size(3) != 0:
                y = model(g, h, edge, numeric,sequence)
                loss = criterion(y, label.to("cuda"))
                # print("Loss:\t", loss)
                epoch_loss += loss.item()
                loss.backward()
                optimiser.step()
            else:
                print('-' * 50)
                print('detected zero dimensional vector - skipping')
                print('-' * 50)

        epoch_loss = epoch_loss / len(train_label)
        print(f"epoch: {e}\tloss: {epoch_loss}")
        tr_acc = test_multi_dropout(model, train_dataloader, criterion)
        res_dict = utils.log_eval_report("train", tr_acc, mlflow, e, res_dict, r)

        test_acc = test_multi_dropout(model, test_dataloader, criterion)
        res_dict = utils.log_eval_report("test", test_acc, mlflow, e, res_dict, r)
        if cfg.train.lr_decay:
            scheduler.step(test_acc["f1-macro"])

        for k1, v1 in tr_acc.items():
            if k1 in [
                    "loss",
                    "f1-macro",
                    "f1-micro",
                    "precision-macro",
                    "recall-macro",
                    "auroc-macro",

                    "var-f1-macro",
                    "var-precision-macro",
                    "var-recall-macro",
                    "var-auroc-macro",
                ]:
                mlflow.log_metric('train-' + k1 +'-in-loop', v1, step=e)
            else:
                mlflow.log_metric('train-' + k1+'-in-loop', v1[-1], step=e)

        for k1, v1 in test_acc.items():
            if k1 in [
                    "loss",
                    "f1-macro",
                    "f1-micro",
                    "precision-macro",
                    "recall-macro",
                    "auroc-macro",

                    "var-f1-macro",
                    "var-precision-macro",
                    "var-recall-macro",
                    "var-auroc-macro",
                ]:
                mlflow.log_metric('test-' + k1+'-in-loop', v1, step=e)
            else:
                mlflow.log_metric('test-' + k1+'-in-loop', v1[-1], step=e)

    return res_dict
    
@hydra.main(version_base="1.2", config_path="config", config_name="config_MULTI.yaml")
def main(cfg):

    hydra_cfg = HydraConfig.get()
    run_name = hydra_cfg.job.config_name
    data_dir = cfg.train.data_dir

    X_train, X_test, y_train, y_test = preprocessing.load_data(
        data_dir, cfg.train.num_run, cfg.train.train_percent
    )
    
    res_dict = defaultdict(dict)

    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name):
        log_params_from_omegaconf_dict(cfg)

        for r in range(cfg.train.start_split, cfg.train.end_split):
            feat_train = X_train[r]
            label_train = y_train[r]

            feat_test = X_test[r]
            label_test = y_test[r]

            if cfg.features.pca:
                feat_train = utils.pca(feat_train, cfg.features.pca_fac)
                feat_test = utils.pca(feat_test, cfg.features.pca_fac)

            train_data = (feat_train, label_train)
            validation_data = (feat_test, label_test)

            #non-pretrained Mordred
            if cfg.train.model_type == 'MLP':
                
                numeric_feature = cfg.train.numeric_feature
                l_target = cfg.train.l_target
                dropout_prob = cfg.train.dropout_prob
                
                model = MLP(numeric_feature,l_target, dropout_prob).to("cuda")

            #non-pretrained Graph
            elif cfg.train.model_type == 'MPNN':
                
                in_n = [cfg.train.in_n1,cfg.train.in_n2]  
                hidden_state_size = cfg.train.hidden_state_size 
                message_size = cfg.train.message_size
                n_layers = cfg.train.n_layers 
                l_target = cfg.train.l_target
                
                model = MPNN(in_n, hidden_state_size, message_size, n_layers, l_target, type="classification").to("cuda")
            #non-pretrained SMILES
            elif cfg.train.model_type == 'TRANS':
                image_size = cfg.train.image_size
                time_size = cfg.train.time_size
                fre_size = cfg.train.fre_size
                dim = cfg.train.dim 
                depth = cfg.train.depth 
                heads = cfg.train.heads
                mlp_dim = cfg.train.mlp_dim 
                dim_head = cfg.train.dim_head
                l_target = cfg.train.l_target
                
                model = TRANS(image_size,time_size,fre_size,l_target, dim, depth, heads, mlp_dim,dim_head).to("cuda")

            #non-pretrained MORDRED + GRAPH + SMILES with MLP head
            elif cfg.train.model_type == 'MULTI':
                image_size = cfg.train.image_size
                time_size = cfg.train.time_size
                fre_size = cfg.train.fre_size
                dim = cfg.train.dim 
                depth = cfg.train.depth 
                heads = cfg.train.heads
                mlp_dim = cfg.train.mlp_dim 
                dim_head = cfg.train.dim_head
                
                in_n = [cfg.train.in_n1,cfg.train.in_n2]  
                hidden_state_size = cfg.train.hidden_state_size 
                message_size = cfg.train.message_size
                n_layers = cfg.train.n_layers 
                
                numeric_feature = cfg.train.numeric_feature
                
                l_target = cfg.train.l_target

                final_op = cfg.train.final_op
                
                model = MULTI(image_size,time_size,fre_size, dim, depth, heads, mlp_dim,dim_head,in_n, hidden_state_size, message_size, n_layers, numeric_feature, l_target, final_op=final_op).to("cuda")

            #non-pretrained GRAPH + SMILES with label regularizer
            elif cfg.train.model_type == 'MPNN_TRANS_LR':
                image_size = cfg.train.image_size
                time_size = cfg.train.time_size
                fre_size = cfg.train.fre_size
                dim = cfg.train.dim 
                depth = cfg.train.depth 
                heads = cfg.train.heads
                mlp_dim = cfg.train.mlp_dim 
                dim_head = cfg.train.dim_head
                
                in_n = [cfg.train.in_n1,cfg.train.in_n2]  
                hidden_state_size = cfg.train.hidden_state_size 
                message_size = cfg.train.message_size
                n_layers = cfg.train.n_layers 
                
                numeric_feature = cfg.train.numeric_feature
                l_target = cfg.train.l_target
                
                model = MPNN_TRANS_LR(image_size,time_size,fre_size, dim, depth, heads, mlp_dim,dim_head,in_n, hidden_state_size, message_size, n_layers, numeric_feature, l_target).to("cuda")
            #pretrained SMILES
            elif cfg.train.model_type == 'EMBED':
                embed_feature = cfg.train.embed_feature
                l_target = cfg.train.l_target
                dropout_prob = cfg.train.dropout_prob
                
                model = MLP(embed_feature,l_target, dropout_prob).to("cuda")
            #pretrained GRAPH
            elif cfg.train.model_type == 'MOLCLR':
                model = GINet(task='classification', num_layer=5, emb_dim=300, feat_dim=512, 
                            drop_ratio=0.3, pool='mean', pred_n_layer=2, pred_act='softplus').to("cuda")

                # checkpoints_folder = os.path.join('./ckpt', 'pretrained_gin', 'checkpoints')
                checkpoints_folder = os.path.join('/home/ubuntu/work/multi-modal/ckpt', 'pretrained_gin', 'checkpoints')
                state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location='cuda')
                # model.load_state_dict(state_dict)
                model.load_my_state_dict(state_dict)
            #pretrained GRAPH + SMILES with MLP head
            elif cfg.train.model_type == 'MOLCLREMBED_MLP':
                l_target = cfg.train.l_target
                embed_feature = cfg.train.embed_feature
                dropout_prob = cfg.train.dropout_prob
                final_op = cfg.train.final_op

                model = MOLCLREMBED_MLP(l_target, embed_feature=512, final_op=final_op, dropout_embed=dropout_prob).to("cuda")
            #pretrained GRAPH + SMILES with label regularizer
            elif cfg.train.model_type == 'MOLCLREMBED_LR':
                embed_feature = cfg.train.embed_feature
                dropout_prob = cfg.train.dropout_prob
                model = MOLCLREMBED_LR(embed_feature=embed_feature, dropout_embed=dropout_prob).to("cuda")

            else:
                sys.exit()

            optimiser = optim.Adam(model.parameters(), lr=cfg.train.lr)

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimiser,
                mode="max",
                factor=0.5,
                patience=150,
                min_lr=0.00001,
                # min_lr=1e-10,
                threshold=0.001,
                threshold_mode="rel",
                verbose=True,
            )

            #non-pretrained MORDRED
            if cfg.train.model_type == 'MLP':
                res_dict = train_numeric(
                    train_data,
                    validation_data,
                    model,
                    optimiser,
                    scheduler,
                    cfg,
                    r,
                    res_dict,
                )
            #non-pretrained GRAPH
            elif cfg.train.model_type == 'MPNN':
                res_dict = train_graph(
                    train_data,
                    validation_data,
                    model,
                    optimiser,
                    scheduler,
                    cfg,
                    r,
                    res_dict,
                )
            #non-pretrained SMILES              
            elif cfg.train.model_type == 'TRANS':
                res_dict = train_sequence(
                    train_data,
                    validation_data,
                    model,
                    optimiser,
                    scheduler,
                    cfg,
                    r,
                    res_dict,
                )
            # non-pretrained GRAPH + SMILES WITH LABEL REGULARIZER
            elif cfg.train.model_type == 'MPNN_TRANS_LR':
                res_dict = train_multi_lr(
                    train_data,
                    validation_data,
                    model,
                    optimiser,
                    scheduler,
                    cfg,
                    r,
                    res_dict,
                )

            # non-pretrained MORDRED + GRAPH + SMILES WITH MLP HEAD
            elif cfg.train.model_type == 'MULTI':
                res_dict = train_multi(
                    train_data,
                    validation_data,
                    model,
                    optimiser,
                    scheduler,
                    cfg,
                    r,
                    res_dict,
                ) 
            # pretrained SMILES
            elif cfg.train.model_type == 'EMBED':
                res_dict = train_embedding(
                    train_data,
                    validation_data,
                    model,
                    optimiser,
                    scheduler,
                    cfg,
                    r,
                    res_dict,
                )
            # pretrained GRAPH
            elif cfg.train.model_type == 'MOLCLR':
                res_dict = train_molclr(
                    train_data,
                    validation_data,
                    model,
                    optimiser,
                    scheduler,
                    cfg,
                    r,
                    res_dict,
                )
            # pretrained GRAPH + SMILES WITH MLP HEAD
            elif cfg.train.model_type == 'MOLCLREMBED_MLP':
                res_dict = train_molclrembed_mlp(
                    train_data,
                    validation_data,
                    model,
                    optimiser,
                    scheduler,
                    cfg,
                    r,
                    res_dict,
                )
            # pretrained GRAPH + SMILES WITH LABEL REGULARIZER
            elif cfg.train.model_type == 'MOLCLREMBED_LR':
                res_dict = train_molclrembed_lr(
                    train_data,
                    validation_data,
                    model,
                    optimiser,
                    scheduler,
                    cfg,
                    r,
                    res_dict,
                )
            else:
                sys.exit()
        # print(res_dict.keys())
        epochs = res_dict.keys()
        for epoch in epochs:
            cur_result = res_dict[epoch]
            for k1, v1 in cur_result.items():
                value = np.mean(list(v1.values()))
                mlflow.log_metric(k1, value, step=epoch)


if __name__ == "__main__":
    experiment = mlflow.set_experiment("final")
    main()
