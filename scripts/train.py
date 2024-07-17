# -*-coding:utf-8-*-
import numpy as np
import esm
import torch.nn as nn
import torch
from torch.utils.data.dataloader import DataLoader
import os
import csv
# from apex import amp

from loading_Data import TrainData, TestData

class LaccaseModel(torch.nn.Module):
    def __init__(self):
        super(LaccaseModel,self).__init__()
        self.modelEsm, alphabet = esm.pretrained.load_model_and_alphabet_local("/zhouxibin/models/esm1b_t33_650M_UR50S.pt")
        self.dnn = nn.Sequential(
            # nn.Linear(1280, 1280),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1280, 2)
        )

    def forward(self,data):
        result = self.modelEsm(data, repr_layers=[33])
        out_result = result["representations"][33][:, 0, :].squeeze()
        out_put = self.dnn(out_result).squeeze()
        return out_put

import argparse

def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--batch_size', type=int, default=16)
    argparser.add_argument('--epoch', type=int, default=1000)
    argparser.add_argument('--last_layers', type=int, default=0)
    args = argparser.parse_args()
    return args
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    args = parse_args()
    BATCH_SIZE = int(args.batch_size)
    EPOCH = int(args.epoch)

    Test_Acc = []

    Test_data = TestData("test_all.txt")
    test_data_dataset = DataLoader(dataset=Test_data, batch_size=100, shuffle=True,
                                    collate_fn=Test_data.collate__fn, drop_last=False, pin_memory=True)

    model = LaccaseModel()
    criterion = torch.nn.CrossEntropyLoss()
    last_layers = int(args.last_layers)

    for name, param in model.named_parameters():
        param.requires_grad = False
        for last_layer in range(1, last_layers+1):
            if f"layers.{33-last_layer}." in name:
                param.requires_grad = True
        if "dnn" in name:
            param.requires_grad = True
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    model = model.cuda()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    # model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

    for epoch in range(EPOCH):
        Train_data = TrainData("train_positive.txt", "train_negative.txt")
        train_data_dataset = DataLoader(dataset=Train_data, batch_size=BATCH_SIZE, shuffle=True,
                                        collate_fn=Train_data.collate__fn, drop_last=True, pin_memory=True)
        # train
        model.train()
        for i, item in enumerate(train_data_dataset):

            content, label = item
            content = content.cuda()
            label = label.cuda()
            last_result = model(content)
            loss = criterion(last_result, label)
            print("epoch: {} \t iteration : {} \t Loss: {} \t lr: {}".format(epoch, i, loss.item(),
                                                                             optimizer.param_groups[0]['lr']), flush=True)

            optimizer.zero_grad()
            # loss.backward
            loss.backward()
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            optimizer.step()
            if i % (len(train_data_dataset)//2) == 0:
                # eval
                model.eval()
                total_test = 0
                correct_test = 0
                predict_test = {}
                predict_really_test = {}
                grand_truth_test = {0:Train_data.all_data_negnative.shape[0],1:Train_data.all_data_positive.shape[0]}
                for m, test in enumerate(test_data_dataset):
                    data_test, label_test = test
                    data_test = data_test.to(device)
                    label_test = label_test.to(device)

                    with torch.no_grad():
                        last_result_test = model(data_test).to(device)

                    # label
                    predicted = torch.argmax(last_result_test.data, dim=1)

                    predict_label = predicted.cpu().numpy()
                    really_label = label_test.cpu().numpy()

                    # predict        是预测出来的label
                    # predict_really 是预测正确的label
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
                for m in range(len(grand_truth_test)):
                    if m in predict_test and m in predict_really_test:
                        out = out + "Category_" + str(m) + "\t" + "predict_really " + str(predict_really_test[m]) + \
                                "\t" + "predict " + str(predict_test[m]) + "\t" + "     Precision " + str(
                                predict_really_test[m] / predict_test[m]) + "\t" \
                                + "  recall" + str(predict_really_test[m] / grand_truth_test[m]) + "\n"
                print("Epoch_item: {} \t\t Correct_num: {} \t\t total: {} \t\t Accuracy on test data: {} \n".format(
                    epoch, correct_test, total_test, correct_test / total_test))
                print(out)
                with open(f"./result/dnn_result_test_lastlayer{last_layers}.txt", "a+", encoding="utf-8") as output:
                    output.write(
                            "Epoch_item: {} \t\t Correct_num: {} \t\t total: {} \t\t Accuracy on test data: {} \n".format(
                                epoch, correct_test, total_test, correct_test / total_test))
                    output.write(out)

                Test_Acc.append(correct_test / total_test)
                with open(f"./result/dnn_result_test_ACC_lastlayer{last_layers}.txt", "a+", encoding="utf-8") as output:
                    output.write(str(Test_Acc) + "\n")
                model.train()
        
        # save model
        save_path = f"./ckpt/dnn_model_lastlayer{last_layers}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), os.path.join(save_path, f"epoch{epoch}.pth"))
