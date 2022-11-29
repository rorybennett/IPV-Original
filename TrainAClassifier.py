import csv
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import parameters as pms
from My_Dataset import My_Dataset
from My_Dataset import ToTensor
from Quadruplet import Quadruplet

for fold_num in range(pms.num_of_folds):
    csv_path = pms.train_path + "/Train_f" + str(fold_num + 1) + ".csv"
    dataset = My_Dataset(csv_path, transform=ToTensor())
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=pms.batch_size, shuffle=True, num_workers=10)
    val_dataset = My_Dataset(pms.train_path + "/Val.csv", transform=ToTensor())
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=pms.batch_size, shuffle=True, num_workers=10)
    log_csv_file = open(pms.train_path + "/TrainLog_f" + str(fold_num + 1) + "_" + pms.train_network + "_bs" + str(
        pms.batch_size) + "_lr" + str(pms.lr) + ".csv", 'w')
    log_csv_writer = csv.writer(log_csv_file)
    log_csv_writer.writerow(["lr", "epoch", "data_num", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])
    net = Quadruplet().cuda()
    print(net)
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=pms.lr, momentum=0.9)
    if pms.lr_schedule:
        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    fin = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    loss_prints = []
    val_acc_check = [0.0, 0.0]
    for epoch in range(15):
        if val_acc_check[1] < val_acc_check[0]:
            if pms.lr_schedule: scheduler.step()
        val_acc_check[0] = val_acc_check[1]
        if fin == 1:
            break
        print(time.localtime().tm_mday, "/", time.localtime().tm_mon, "/", time.localtime().tm_year, " ",
              time.localtime().tm_hour, ":", time.localtime().tm_min)
        running_loss = 0.0
        batch_count = 0
        train_losses.append(0)
        train_accuracies.append(0)
        val_losses.append(0)
        val_accuracies.append(0)
        loss_prints.append(epoch + 2)
        for i, data in enumerate(data_loader, 0):
            batch_count += 1
            image = data['image'].cuda()
            labels = data['labels'].cuda()
            optimizer.zero_grad()
            outs = net(image)
            train_loss = 0
            train_count = 0
            train_accuracy = 0
            for labs in range(len(pms.num_of_classes)):
                t_l = criterion(outs[labs], labels[:, labs])
                train_loss += t_l
                for t_i, t_out in enumerate(outs[labs]):
                    train_count += 1
                    if torch.argmax(t_out) == labels[t_i][labs]:
                        train_accuracy += 1
            train_accuracy /= train_count
            train_accuracies[-1] += train_accuracy
            train_loss /= len(pms.num_of_classes)
            train_loss.backward()
            optimizer.step()
            running_loss += train_loss.item()
            train_losses[-1] += train_loss.item()
            if batch_count % pms.loss_print == 0 or (epoch == 0 and i == 0):
                # validation
                net.eval()
                with torch.no_grad():
                    all_count = 0
                    val_accuracy = 0
                    val_loss = 0
                    val_count = 0
                    for val_i, val_data in enumerate(val_data_loader, 0):
                        val_image = val_data['image'].cuda()
                        val_labels = val_data['labels'].cuda()
                        val_out = net(val_image)
                        for p_i, point_outs in enumerate(val_out):
                            for cl_i, cl_out in enumerate(point_outs):
                                all_count += 1
                                if torch.argmax(cl_out) == val_labels[cl_i][p_i]:
                                    val_accuracy += 1
                        v_l = 0
                        for labs in range(len(pms.num_of_classes)):
                            v_l += criterion(val_out[labs], val_labels[:, labs])
                        v_l /= len(pms.num_of_classes)
                        val_loss += v_l.item()
                        val_count += 1
                    val_accuracy /= all_count
                    val_loss = val_loss / val_count
                net.train()
                # validation
                val_losses[-1] = val_loss
                val_accuracies[-1] = val_accuracy
                if (epoch == 0 and i == 0):
                    train_losses.append(0)
                    train_accuracies.append(0)
                    val_losses.append(0)
                    val_accuracies.append(0)
                    loss_prints[-1] = 0
                    loss_prints.append(1)
                else:
                    running_loss = running_loss / pms.loss_print
                    if running_loss < 0.01:
                        fin = 1
                val_acc_check[1] = val_accuracy
                print('[epoch: %d, data_num: %5d] train_loss: %f, train_accuracy: %f, val_loss: %f, val_accuracy: %f' %
                      (epoch + 1, batch_count * pms.batch_size, running_loss, train_accuracy, val_loss, val_accuracy))
                log_csv_writer.writerow(
                    [str(optimizer.param_groups[0]['lr']), str(epoch + 1), str(batch_count * pms.batch_size),
                     str(running_loss), str(train_accuracy), str(val_loss), str(val_accuracy)])

                running_loss = 0.0
        torch.save(net.state_dict(),
                   pms.train_path + "/" + pms.train_network + "_bs" + str(pms.batch_size) + "_curr_state_f" + str(
                       fold_num + 1) + "_lr" + str(pms.lr) + "_e" + str(epoch + 1) + ".pth")
        train_accuracies[-1] /= batch_count
        train_losses[-1] /= batch_count
        plt.plot(loss_prints, train_losses, 'r--')
        plt.plot(loss_prints, train_accuracies, 'r-')
        plt.plot(loss_prints, val_losses, 'b--')
        plt.plot(loss_prints, val_accuracies, 'b-')
        plt.legend(['Training Loss', 'Train accuracy', 'Val Loss', 'Val accuracy'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(pms.train_path + "/f" + str(fold_num + 1) + "_" + pms.train_network + "_bs" + str(
            pms.batch_size) + "_lr" + str(pms.lr) + ".png")
    print('Finished training!')
    plt.clf()
