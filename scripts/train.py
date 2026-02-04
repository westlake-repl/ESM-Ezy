# -*-coding:utf-8-*-
import numpy as np
import esm
import torch.nn as nn
import torch
from torch.utils.data.dataloader import DataLoader
import os
import csv
# from apex import amp

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from loading_Data import TrainData, TestData
from model import LaccaseModel
from dataset import TrainingDataset
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_positive_data', type=str)
    parser.add_argument('--train_negative_data', type=str)
    parser.add_argument('--test_positive_data', type=str)
    parser.add_argument('--test_negative_data', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--last_layers', type=int, default=1)
    parser.add_argument('--save_path', type=str, default=".")
    parser.add_argument('--embedding_source', type=str, default="fair_esm2", choices=["fair_esm2", "precomputed"])
    parser.add_argument('--esm_checkpoint', type=str, default="esm2_t36_3B_UR50D")
    parser.add_argument('--device', type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument('--precision', type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument('--precomputed_embeddings_dir', type=str, default=None)
    parser.add_argument('--precomputed_format', type=str, default="auto", choices=["auto", "npy", "npz", "pt"])
    parser.add_argument('--precomputed_granularity', type=str, default="per_sequence",
                        choices=["per_sequence", "per_residue"])
    # Add Early Stop parameter with default=None to indicate disabled
    parser.add_argument('--patience', type=int, default=None, help='Number of epochs to wait before early stop. If not provided, early stop is disabled.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = parse_args()
    train_positive_data = args.train_positive_data
    train_negative_data = args.train_negative_data
    test_positive_data = args.test_positive_data
    test_negative_data = args.test_negative_data
    model_path = args.model_path
    esm_checkpoint = args.esm_checkpoint if model_path is None else model_path
    BATCH_SIZE = int(args.batch_size)
    EPOCH = int(args.epoch)
    last_layers = int(args.last_layers)
    total_save_path = args.save_path
    patience = args.patience  # None if not provided
    
    # ==================== Early Stop Initialization ====================
    if patience is not None:
        # Early stop is enabled when patience is provided
        best_acc = 0.0           # Record the best accuracy
        best_epoch = -1          # Record the epoch where best accuracy occurred
        wait_counter = 0         # Counter for no improvement
        print(f"Early stop enabled with patience={patience}")
    else:
        print("Early stop disabled")
    
    # model
    print("Loading model...")
    model = LaccaseModel(
        pretrained_model_path=esm_checkpoint,
        embedding_source=args.embedding_source,
        esm_checkpoint=esm_checkpoint,
        device=args.device,
        precision=args.precision,
        precomputed_embeddings_dir=args.precomputed_embeddings_dir,
        precomputed_format=args.precomputed_format,
        precomputed_granularity=args.precomputed_granularity,
    )
    model.set_trainable_last_layers(last_layers)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    model = model.to(model.device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    # data
    train_dataset = TrainingDataset(positive_path=train_positive_data, negative_path=train_negative_data, dynamic_negative_sampling=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                    collate_fn=train_dataset.collate_fn, drop_last=False, pin_memory=True)
    test_dataset = TrainingDataset(positive_path=test_positive_data, negative_path=test_negative_data)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                    collate_fn=test_dataset.collate_fn, drop_last=False, pin_memory=True)

    Test_Acc = []
    for epoch in range(EPOCH):
        # ==================== Early Stop Check ====================
        if patience is not None and wait_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}!")
            print(f"Best test accuracy {best_acc:.4f} achieved at epoch {best_epoch}")
            break
        
        # train
        model.train()
        print(len(train_dataloader))
        for i, item in enumerate(train_dataloader):
            label = item["labels"].to(model.device)
            last_result = model(item)
            loss = criterion(last_result, label)
            print("epoch: {} \t iteration : {} \t Loss: {} \t lr: {}".format(epoch, i, loss.item(),
                                                                             optimizer.param_groups[0]['lr']), flush=True)
            # break
            optimizer.zero_grad()
            # loss.backward
            loss.backward()
            optimizer.step()
            if i % (len(train_dataloader)//2) == 0:
                # eval
                model.eval()
                total_test = 0
                correct_test = 0
                predict_test = {}
                predict_really_test = {}
                ground_truth_test = {0:len(test_dataset.negative_dataset),1:len(test_dataset.positive_dataset)}
                for m, test in enumerate(tqdm(test_dataloader)):
                    label_test = test["labels"].to(model.device)

                    with torch.no_grad():
                        last_result_test = model(test)

                    # label
                    predicted = torch.argmax(last_result_test.data, dim=1)

                    predict_label = predicted.cpu().numpy()
                    really_label = label_test.cpu().numpy()

                    for k in range(len(predict_label)):
                        if predict_label[k] not in predict_test:
                            predict_test[predict_label[k]] = 1
                        else:
                            predict_test[predict_label[k]] += 1

                        if predict_label[k] == really_label[k]:
                            if predict_label[k] not in predict_really_test:
                                predict_really_test[predict_label[k]] = 1
                            else:
                                predict_really_test[predict_label[k]] += 1

                    total_test += label_test.size(0)
                    correct_test += (predicted == label_test).sum().cpu().item()


                out = ""
                for m in range(len(ground_truth_test)):
                    if m in predict_test and m in predict_really_test:
                        out = out + "Category_" + str(m) + "\t" + "predict_really " + str(predict_really_test[m]) + \
                                "\t" + "predict " + str(predict_test[m]) + "\t" + "     Precision " + str(
                                predict_really_test[m] / predict_test[m]) + "\t" \
                                + "recall" + str(predict_really_test[m] / ground_truth_test[m])[:5] + "\n"
                
                current_acc = correct_test / total_test
                Test_Acc.append(current_acc)
                print("Epoch_item: {} \t\t Correct_num: {} \t\t total: {} \t\t Accuracy on test data: {} \n".format(
                    epoch, correct_test, total_test, current_acc))
                print(out)
                result_dir = f"{total_save_path}/result"
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
                with open(f"{result_dir}/dnn_result_test_lastlayer{last_layers}.txt", "a+", encoding="utf-8") as output:
                    output.write(
                            "Epoch_item: {} \t\t Correct_num: {} \t\t total: {} \t\t Accuracy on test data: {} \n".format(
                                epoch, correct_test, total_test, current_acc))
                    output.write(out)

                with open(f"{result_dir}/dnn_result_test_ACC_lastlayer{last_layers}.txt", "a+", encoding="utf-8") as output:
                    output.write(str(Test_Acc) + "\n")
                    
                # ==================== Early Stop Update ====================
                if patience is not None:
                    if current_acc > best_acc:
                        best_acc = current_acc
                        best_epoch = epoch
                        wait_counter = 0  # Reset counter
                        print(f"Test accuracy improved to {best_acc:.4f} at epoch {best_epoch}")
                    else:
                        wait_counter += 1
                        print(f"Test accuracy not improved. Patience: {wait_counter}/{patience}")
                    
                model.train()
        
        # save model
        save_path = f"{total_save_path}/ckpt/dnn_model_lastlayer{last_layers}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), os.path.join(save_path, f"epoch{epoch}.pth"))
        
    # ==================== Final Report ====================
    if patience is not None:
        print(f"\nTraining completed with early stop! Best test accuracy: {best_acc:.4f} at epoch {best_epoch}")
    else:
        print("\nTraining completed without early stop")
