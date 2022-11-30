import csv
import datetime as dt
import os

import cv2
import numpy as np
from scipy.spatial import distance as dist
from skimage import img_as_ubyte
from skimage import io
from skimage.transform import resize

import parameters as pms
from GPU_Utils import *


class PatchCreator:

    def __init__(self, image, sub_patch_scales):
        self.sub_patch_scales = sub_patch_scales
        self.patch_size = sub_patch_scales[0]
        self.image = image

    def create(self, x, y):

        sub_patches = []
        for scale in self.sub_patch_scales:
            sub_patch = createPatch(self.image, x, y, scale)
            sub_patches.append(sub_patch)

        arr_patches = []
        for i in range(4):
            arr_patches.append(resize(sub_patches[i], (self.patch_size, self.patch_size), preserve_range=True))

        return arr_patches


class DataCreator:

    def __init__(self, distance_intervals, angle_intervals, folds, subpatch_scales):

        self.current_fold = None
        self.patch_count = None
        self.pc = None
        self.labels_count = None
        self.current_sample_name = None
        self.show_image = None
        self.name_count = None
        self.csv_files = None
        self.save_path = None
        self.sub_patch_scales = subpatch_scales
        self.patch_size = subpatch_scales[0] * 2
        self.folds = folds
        self.distance_intervals = distance_intervals
        self.angle_intervals = angle_intervals
        self.points_dict = {}  # -------> [train] or [test]
        self.paths_dict = {}  # --------> [train] or [test]
        self.fold_list = []  # -----> [[[train],[test]], [[train],[test]], ...]
        self.pix_to_cm_vals_dict = {}

    # Create lists of names for each fold to self.fold_list
    def read_fold_lists(self):
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
            self.fold_list.append([train_list, test_list, val_list])
            for trl in train_list:
                for tel in test_list:
                    for vl in val_list:
                        if trl == tel or trl == vl or tel == vl:
                            print("Error in folds!")
                            exit()
        print("Folds read.")

    # Read points and paths of a given names list to self.points and self.paths
    def read_points(self, phase):
        if phase == "Train":
            names_list = self.fold_list[self.current_fold - 1][0]
        elif phase == "Test":
            names_list = self.fold_list[self.current_fold - 1][1]
        elif phase == "Val":
            names_list = self.fold_list[self.current_fold - 1][2]
        else:
            names_list = []
        self.points_dict = {}
        self.paths_dict = {}
        all_paths_dict = {}
        lines_dict = {}
        mark_list_file = open(pms.resampled_mark_list_path, "r")
        for ml in mark_list_file:
            path = ml[0:ml.find(" ")]
            name = path[0:path.find(".")]
            all_paths_dict[name] = path
            lines_dict[name] = ml
        for list_name in names_list:
            print(all_paths_dict[list_name])
            self.paths_dict[list_name] = pms.resampled_data_path + '/' + all_paths_dict[list_name]
            pt = []
            num = lines_dict[list_name]
            for i in range(pms.num_of_pts):
                num = num[num.find(" (") + 1:len(num)]
                p = (int(num[1:num.find(",")]), int(num[num.find(" ") + 1:num.find(")")]))
                pt.append(p)
            print(pt)
            self.points_dict[list_name] = pt

    def create_csv(self, phase):
        csvfile1 = open(pms.data_path + "/" + phase + "_f" + str(self.current_fold) + ".csv", 'w')
        data_file = csv.writer(csvfile1)
        return [data_file]

    def create_x_y(self, x, y):
        labels = []
        x = int(x)
        y = int(y)
        for i, p in enumerate(self.points_dict[self.current_sample_name][0:pms.num_of_pts]):
            pix_distance = dist.euclidean(p, (x, y))
            dist_class = get_label(pix_distance, self.distance_intervals)
            ang_class = get_label(int(getAngle(p, (x, y))), self.angle_intervals)
            labels.append(dist_class)
            labels.append(ang_class)
            self.labels_count[i][dist_class] += 1
        self.patch_count += 1
        created = self.pc.create(x, y)
        for img_num, created_image in enumerate(created):
            image = img_as_ubyte(created_image)
            curr_save_path = self.save_path + "/" + str(self.name_count) + "_" + str(x) + "_" + str(y) + "_" + str(
                img_num + 1) + ".png"
            io.imsave(curr_save_path, image)
            if pms.num_of_pts == 2:
                self.csv_files[0].writerow([str(self.name_count) + "_" + str(x) + "_" + str(y),
                                            curr_save_path, self.current_sample_name, x, y,
                                            labels[0], labels[1], labels[2], labels[3]])
            else:
                self.csv_files[0].writerow([str(self.name_count) + "_" + str(x) + "_" + str(y),
                                            curr_save_path, self.current_sample_name, x, y,
                                            labels[0], labels[1], labels[2], labels[3], labels[4], labels[5], labels[6],
                                            labels[7]])
        cv2.circle(self.show_image, (x, y), 1, (255, 0, 0), 1)
        # print(x, y)
        # io.imshow(self.show_image)
        # io.show()
        # io.imshow(created)
        # io.show()

    # Create and save patches with labels
    def create_data(self, step, phase):
        self.read_points(phase)
        self.save_path = pms.data_path + "/" + phase + "_f" + str(self.current_fold)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(pms.data_path + "/" + phase + "_images/"):
            os.makedirs(pms.data_path + "/" + phase + "_images/")
        self.csv_files = self.create_csv(phase)
        # self.read_cm_to_pix_vals()

        # create patches with labels
        self.name_count = 0
        for name in self.points_dict:  # four points of a sample
            self.name_count += 1
            data_path = self.paths_dict[name]
            image = io.imread(data_path, as_gray=True)
            self.show_image = io.imread(data_path)
            print(data_path)
            self.current_sample_name = name
            if pms.num_of_pts == 2:
                self.labels_count = [numpy.zeros(len(self.distance_intervals), dtype=numpy.int64),
                                     numpy.zeros(len(self.distance_intervals), dtype=numpy.int64)]
            else:
                self.labels_count = [numpy.zeros(len(self.distance_intervals), dtype=numpy.int64),
                                     numpy.zeros(len(self.distance_intervals), dtype=numpy.int64),
                                     numpy.zeros(len(self.distance_intervals), dtype=numpy.int64),
                                     numpy.zeros(len(self.distance_intervals), dtype=numpy.int64)]

            self.pc = PatchCreator(image, sub_patch_scales=self.sub_patch_scales)
            self.patch_count = 0
            time1 = dt.datetime.now()

            if phase == "Test":
                for x in range(0, image.shape[1], step):
                    for y in range(0, image.shape[0], step):
                        self.create_x_y(x, y)
            elif phase == "Train" or phase == "Val":
                for pt in range(pms.num_of_pts):
                    for ran in range(2):
                        for num in range(pms.patches_per_sample // 8):
                            if ran == 0:
                                x, y = np.random.multivariate_normal(
                                    (self.points_dict[name][pt][0], self.points_dict[name][pt][1]),
                                    [[500, 0], [0, 500]]).T
                            else:
                                x, y = np.random.multivariate_normal(
                                    (self.points_dict[name][pt][0], self.points_dict[name][pt][1]),
                                    [[10000, 0], [0, 10000]]).T
                            if x > image.shape[1]:
                                x = x - image.shape[1]
                            if y > image.shape[0]:
                                y = y - image.shape[0]
                            if x < 0:
                                x = x + image.shape[1]
                            if y < 0:
                                y = y + image.shape[0]
                            self.create_x_y(x, y)

            print(self.labels_count)
            print("Labels sum: ", np.sum(self.labels_count))
            print("Patch count: ", self.patch_count)
            for pt in range(pms.num_of_pts):
                cv2.circle(self.show_image, self.points_dict[name][pt], 1, (255, 255, 255), 1)
            io.imsave(pms.data_path + "/" + phase + "_images/" + self.current_sample_name + ".png", self.show_image)
            time2 = dt.datetime.now()
            print(time1 - time2)
        # create patches with labels

    # phase = train, test, both
    def create(self, step, phase, current_fold):
        self.current_fold = current_fold
        if not self.fold_list:
            self.read_fold_lists()
        if phase == 'both':
            self.create_data(step, "Train")
            if self.current_fold == 1:
                self.create_data(step, "Val")
            self.create_data(step, "Test")
        else:
            self.create_data(step, phase)


# -------------------------------------------------------------------------------------------------------------------
# RUN
start_time = dt.datetime.now()

dc = DataCreator(distance_intervals=pms.tasks_classes[0], angle_intervals=pms.tasks_classes[1], folds=pms.num_of_folds,
                 subpatch_scales=pms.sub_patch_scales)
for f in range(pms.num_of_folds):
    dc.create(step=pms.test_data_step, phase=pms.phase, current_fold=f + 1)
    infocsv = open(pms.data_path + "/data_info_" + pms.phase + ".csv", "w")
    info_file = csv.writer(infocsv)
    info_file.writerow(
        ["DISTANCE_INTERVALS", "ANGLE_INTERVALS", "NUM_OF_FOLDS", "HAVE_SUB_PATCHES", "SUB_PATCH_SCALES", "PATCH_SIZE"])
    info_file.writerow(
        [str(pms.tasks_classes[0]), str(pms.tasks_classes[1]), pms.num_of_folds, pms.sub_patches, pms.sub_patch_scales,
         pms.sub_patch_scales[0]])
    infocsv.close()

end_time = dt.datetime.now()
print("Finished in:", end_time - start_time)
