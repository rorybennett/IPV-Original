import csv
import os

import cv2
import numpy as np
import torch
from scipy import ndimage as nd
from scipy.spatial import distance as dist
from skimage import io

import parameters as pms
from My_Dataset import My_Dataset
from My_Dataset import ToTensor
from Quadruplet import Quadruplet


# Read cm to pixel values
# def read_cm_to_pix_vals():
#     pix_to_cm_vals_dict = {}
#     f = open(pms.cm_to_pix_path, 'r')
#     for line in f:
#         name = line[0: line.find(".")]
#         val = line[line.find(" ") + 1: len(line) - 1]
#         pix_to_cm_vals_dict[name] = float(val)
#     return pix_to_cm_vals_dict

def readFoldLists(mark_list_path):
    fold_list = []
    for i in range(1, pms.num_of_folds + 1):
        train_list = []
        test_list = []
        val_list = []
        train_file = open(pms.fold_lists_path + "/train_f" + str(i) + ".txt")
        for no in train_file:
            train_list.append(no[0:-1])
        test_file = open(pms.fold_lists_path + "/test_f" + str(i) + ".txt")
        for no in test_file:
            test_list.append(no[0:-1])
        val_file = open(pms.fold_lists_path + "/val.txt")
        for no in val_file:
            val_list.append(no[0:-1])
        fold_list.append([train_list, test_list, val_list])
        for trl in train_list:
            for tel in test_list:
                for vl in val_list:
                    if trl == tel or trl == vl or tel == vl:
                        print("Error in folds!")
                        exit()
    print("Folds read.")
    return fold_list


# Read points and paths of a given names list to self.points and self.paths
def readPoints(names_list):
    points_dict = {}
    paths_dict = {}
    all_paths_dict = {}
    lines_dict = {}
    mark_list_file = open(pms.resampled_mark_list_path, "r")
    for ml in mark_list_file:
        path = ml[0:ml.find(" ")]
        name = path[0:path.find(".")]
        all_paths_dict[name] = path
        lines_dict[name] = ml
    for listname in names_list:
        print(all_paths_dict[listname])
        paths_dict[listname] = pms.resampled_data_path + '/' + all_paths_dict[listname]
        pt = []
        num = lines_dict[listname]
        for i in range(pms.num_of_pts):
            num = num[num.find(" (") + 1:len(num)]
            p = (int(num[1:num.find(",")]), int(num[num.find(" ") + 1:num.find(")")]))
            pt.append(p)
        print(pt)
        points_dict[listname] = pt
    return points_dict, paths_dict


# def readMyPoints(names_list):
#     points_dict = {}
#     paths_dict = {}
#     all_paths_dict = {}
#     lines_dict = {}
#     mark_list_file = open(pms.new_doc_1_MarkListPath, "r")
#     for ml in mark_list_file:
#         path = ml[0:ml.find(" ")]
#         name = path[0:path.find(".")]
#         all_paths_dict[name] = path
#         lines_dict[name] = ml
#     for listname in names_list:
#         print(all_paths_dict[listname])
#         paths_dict[listname] = pms.resampled_data_path + '/' + all_paths_dict[listname]
#         pt = []
#         num = lines_dict[listname]
#         for i in range(5):
#             num = num[num.find(" (") + 1:len(num)]
#             p = (int(num[1:num.find(",")]), int(num[num.find(" ") + 1:num.find(")")]))
#             pt.append(p)
#         print(pt)
#         points_dict[listname] = pt
#     return points_dict, paths_dict

if __name__ == '__main__':
    if not os.path.exists(pms.map_creation_path):
        os.makedirs(pms.map_creation_path)
    fold_list = readFoldLists(pms.resampled_mark_list_path)
    for fold in range(1, pms.num_of_folds + 1):
        print("---Fold " + str(fold) + "---")
        points_dict, paths_dict = readPoints(fold_list[fold - 1][1])
        # newdoc_1_points_dict, newdoc_1_paths_dict = readMyPoints(fold_list[fold - 1][1])
        dclassnum = len(pms.tasks_classes[0])
        aclassnum = len(pms.tasks_classes[1])
        dataset = My_Dataset(pms.data_path + "/Test_f" + str(fold) + '.csv', ToTensor())
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        net = Quadruplet().cuda()
        net.load_state_dict(torch.load(pms.model_paths[fold - 1]))
        net.eval()
        csvfile = open(pms.map_creation_path + "/Results_f" + str(fold) + ".csv", 'w')
        results_writer = csv.writer(csvfile)
        results_writer.writerow(
            ["Name", "P1_distance", "P2_distance", "P3_distance", "P4_distance", "Mean_distance", "h_difference",
             "w_difference"])
        csvfile2 = open(pms.map_creation_path + "/DetectedPoints_f" + str(fold) + ".csv", 'w')
        detected_points_writer = csv.writer(csvfile2)
        csvfile3 = open(pms.map_creation_path + "/PatchBasedResults_f" + str(fold) + ".csv", 'w')
        pbr_writer = csv.writer(csvfile3)
        prev_sample_name = ""
        sample_image = []
        for i, data in enumerate(data_loader, 0):
            image = data["image"].cuda()
            sample_name = data["sample_name"][0]
            coordinates = data["coordinates"][0].numpy()
            labels = data["labels"][0].numpy()
            map_results_path = "/MapWithAngleResults_f" + str(fold) + "_s1"
            # print("sample name:", sample_name, " coordinates:", coordinates)
            if sample_name != prev_sample_name or i + 1 == len(data_loader.dataset):
                if sample_image != []:
                    if not os.path.exists(pms.map_creation_path + map_results_path):
                        os.makedirs(pms.map_creation_path + map_results_path)
                    distances = []
                    detected_points = []
                    for s in range(pms.num_of_pts):
                        convolved_map = nd.gaussian_filter(maps[s], 5)
                        max_i = np.argmax(convolved_map) + 1
                        max_y = (max_i // convolved_map.shape[1]) + 1
                        max_x = max_i % convolved_map.shape[1]
                        cv2.circle(sample_image, (max_x, max_y), 1, (0, 0, 0), 2)
                        cv2.circle(sample_image2, (max_x, max_y), 1, (255, 0, 0), 2)
                        cv2.circle(sample_image2,
                                   (points_dict[prev_sample_name][s][0], points_dict[prev_sample_name][s][1]), 1,
                                   (0, 255, 0), 2)
                        # cv2.circle(sample_image2, (newdoc_1_points_dict[prev_sample_name][s][0], newdoc_1_points_dict[prev_sample_name][s][1]), 1, (0, 0, 255), 2)
                        convolved_map = cv2.normalize(convolved_map, None, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)
                        io.imsave(
                            pms.map_creation_path + map_results_path + "/" + prev_sample_name + "_convolved_map" + str(
                                s) + ".png", np.array(convolved_map, dtype=np.uint8))
                        distances.append(dist.euclidean((max_x, max_y), (
                        points_dict[prev_sample_name][s][0], points_dict[prev_sample_name][s][1])))
                        detected_points.append((max_x, max_y))
                    io.imsave(pms.map_creation_path + map_results_path + "/" + prev_sample_name + ".png", sample_image)
                    io.imsave(pms.map_creation_path + map_results_path + "/" + prev_sample_name + "_2.png",
                              sample_image2)
                    if pms.num_of_pts == 4:
                        real_w = dist.euclidean(points_dict[prev_sample_name][1], points_dict[prev_sample_name][3])
                        real_h = dist.euclidean(points_dict[prev_sample_name][0], points_dict[prev_sample_name][2])
                        detected_w = dist.euclidean(detected_points[1], detected_points[3])
                        detected_h = dist.euclidean(detected_points[0], detected_points[2])
                        diff_w = abs(real_w - detected_w)
                        diff_h = abs(real_h - detected_h)
                        results_writer.writerow(
                            [prev_sample_name, distances[0], distances[1], distances[2], distances[3],
                             np.mean(distances), diff_h, diff_w])
                        detected_points_writer.writerow(
                            [prev_sample_name, detected_points[0][0], detected_points[0][1], detected_points[1][0],
                             detected_points[1][1], detected_points[2][0], detected_points[2][1],
                             detected_points[3][0], detected_points[3][1]])
                    else:
                        real_h = dist.euclidean(points_dict[prev_sample_name][0], points_dict[prev_sample_name][1])
                        detected_h = dist.euclidean(detected_points[0], detected_points[1])
                        diff_h = abs(real_h - detected_h)
                        results_writer.writerow(
                            [prev_sample_name, distances[0], distances[1], np.mean(distances), diff_h])
                        detected_points_writer.writerow(
                            [prev_sample_name, detected_points[0][0], detected_points[0][1], detected_points[1][0],
                             detected_points[1][1]])

                    for pvals in TP_TN_FP_FN:
                        for cvals in pvals:
                            pbr_writer.writerow([prev_sample_name, cvals])
                    print(prev_sample_name + " saved")
                prev_sample_name = sample_name
                sample_image = io.imread(pms.resampled_data_path + "/" + sample_name + ".jpg")
                sample_image2 = io.imread(pms.resampled_data_path + "/" + sample_name + ".jpg")
                maps = [np.zeros(sample_image.shape[0:2], np.uint64),
                        np.zeros(sample_image.shape[0:2], np.uint64),
                        np.zeros(sample_image.shape[0:2], np.uint64),
                        np.zeros(sample_image.shape[0:2], np.uint64)]

                TP_TN_FP_FN = np.zeros((pms.num_of_pts, len(pms.tasks_classes[0]), len(pms.tasks_classes[0])))

            outs = net(image)
            ei = 0
            mno = -1
            # print(outs)
            for out, label in zip(outs, labels):
                ei += 1
                attended_classes = 7
                if ei % 2 == 0 and ei < 9:
                    mno += 1
                    res = torch.argsort(prev_out, descending=True).detach().cpu().numpy()[0]
                    res_a = torch.argsort(out, descending=True).detach().cpu().numpy()[0]
                    TP_TN_FP_FN[mno][prev_label][res[0]] += 1
                    temp_map = np.zeros(sample_image.shape[0:2], np.uint8)
                    if res[0] <= len(pms.tasks_classes[0]) - attended_classes and abs(res[0] - res[1]) == 1 and abs(
                            res[0] - res[2]) == 1 and abs(res_a[0] - res_a[1]) == 1 and abs(res_a[0] - res_a[2]) == 1:
                        angle = (res_a[0] + 4) % 8
                        start_angle = pms.tasks_classes[1][angle][0]
                        end_angle = pms.tasks_classes[1][angle][1]
                        #
                        # print(prev_out)
                        # print(res)
                        # print(out)
                        # # print(res_angle)
                        # print(angle)

                        for r_i in range(pms.layers_to_vote):
                            # if r_i != 0 and abs(res[r_i] - res[0]) != 1:
                            #     continue
                            if pms.weighted_map:
                                vote_val = int(prev_out[0][res[r_i]])
                            else:
                                vote_val = 1
                            rad = (pms.tasks_classes[0][res[r_i]][0] + (
                                        pms.tasks_classes[0][res[r_i]][1] - pms.tasks_classes[0][res[r_i]][0]) / 2)
                            th = (pms.tasks_classes[0][res[r_i]][1] - pms.tasks_classes[0][res[r_i]][0])
                            cv2.ellipse(temp_map, (coordinates[0], coordinates[1]), (int(rad), int(rad)), 0,
                                        start_angle, end_angle, vote_val, thickness=int(th))
                            cv2.ellipse(sample_image, (coordinates[0], coordinates[1]), (int(rad), int(rad)), 0,
                                        start_angle, end_angle, (255, 0, 0), thickness=1)

                        maps[mno] += temp_map

                        # io.imshow(maps[mno])
                        # io.show()
                prev_out = out
                prev_label = label
                if i == dataset.__len__() - 2:
                    sample_name = "Finish"
