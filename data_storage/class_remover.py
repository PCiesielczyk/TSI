import os
import logging
import shutil
import csv
import argparse

logging.basicConfig(level=logging.DEBUG)

project_dir = os.path.dirname(os.getcwd())
archive_dir = os.path.join(project_dir, 'archive')
meta_dir = os.path.join(archive_dir, 'Meta')
test_dir = os.path.join(archive_dir, 'Test')
train_dir = os.path.join(archive_dir, 'Train')


def remove_file(filename: str):
    if os.path.exists(filename):
        os.remove(filename)
        logging.debug(f'Removed {filename}')
    else:
        logging.warning(f'No such file: {filename}')


def rename_meta_images(class_no: int):
    remove_file(os.path.join(meta_dir, str(class_no) + '.png'))

    dir_list = os.listdir(meta_dir)
    sorted_dir_list = sorted(dir_list, key=lambda d: int(d.split('.')[0]))
    for file in sorted_dir_list:
        file_path = os.path.join(meta_dir, file)
        filename_numeric = file.split('.')[0]
        if os.path.isfile(file_path):
            if int(filename_numeric) > class_no:
                renamed_file_name = os.path.join(meta_dir, str(int(filename_numeric) - 1) + '.png')
                os.rename(file_path, renamed_file_name)
    logging.info(f'Renamed {meta_dir} files')


def remove_meta(class_no: int):
    meta_csv_path = os.path.join(archive_dir, 'Meta.csv')

    with open(meta_csv_path, 'r') as meta_csv:
        reader = csv.reader(meta_csv)
        rows = list(reader)
        new_rows = []

        meta_path_val = 'Meta/' + str(class_no) + '.png'

        for row_index, row in enumerate(rows):
            if row[0] == meta_path_val:
                continue
            elif row_index > 0 and int(row[1]) > class_no:
                decremented_class_no = str(int(row[1]) - 1)
                row[0] = 'Meta/' + decremented_class_no + '.png'
                row[1] = decremented_class_no
            new_rows.append(row)

    with open(meta_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(new_rows)
        logging.info('Processed Meta.csv file')


def rename_train_dirs(class_no: int):
    dir_list = os.listdir(train_dir)
    sorted_dir_list = sorted(dir_list, key=lambda d: int(d))
    for directory in sorted_dir_list:
        dir_path = os.path.join(train_dir, directory)
        if os.path.isdir(dir_path):
            if int(directory) > class_no:
                renamed_dir_name = os.path.join(train_dir, str(int(directory) - 1))
                os.rename(dir_path, renamed_dir_name)
    logging.info(f'Renamed {train_dir} directory')


def remove_training_data(class_no: int):
    class_training_dir = os.path.join(train_dir, str(class_no))

    try:
        shutil.rmtree(class_training_dir)
        logging.info(f'Successfully removed directory {class_training_dir} with all its content')
    except OSError as e:
        logging.error(f'Error occurred while deleting {class_training_dir} directory: {e}')


def rename_test_csv(class_no: int):
    test_csv_path = os.path.join(archive_dir, 'Test.csv')

    with open(test_csv_path, 'r') as test_csv:
        reader = csv.reader(test_csv)
        rows = list(reader)
        new_rows = []

        for row_index, row in enumerate(rows):
            if row[6] == str(class_no):
                test_filename = os.path.join(test_dir, row[7].split('/')[-1])
                remove_file(test_filename)
                continue
            elif row_index > 0 and int(row[6]) > class_no:
                row[6] = str(int(row[6]) - 1)
            new_rows.append(row)

    with open(test_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(new_rows)
        logging.info('Processed Test.csv file')


def rename_train_csv(class_no: int):
    train_csv_path = os.path.join(archive_dir, 'Train.csv')

    with open(train_csv_path, 'r') as train_csv:
        reader = csv.reader(train_csv)
        rows = list(reader)
        new_rows = []

        for row_index, row in enumerate(rows):
            if row[6] == str(class_no):
                continue
            elif row_index > 0 and int(row[6]) > class_no:
                row[6] = str(int(row[6]) - 1)
                image_path_segments = row[7].split('/')
                new_path = image_path_segments[0] + '/' + row[6] + '/' + image_path_segments[2]
                row[7] = new_path
            new_rows.append(row)

    with open(train_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(new_rows)
        logging.info('Processed Train.csv file')


def remove_class_data(class_no: int):
    remove_meta(class_no)
    rename_meta_images(class_no)
    remove_training_data(class_no)
    rename_train_dirs(class_no)
    rename_test_csv(class_no)
    rename_train_csv(class_no)


parser = argparse.ArgumentParser(description="Remove class from Meta, Test and Train data.")
parser.add_argument("--class_no", help="No. of class to be removed")
arg = parser.parse_args()

if arg.class_no is None:
    logging.error("Argument --class_no not specified")
elif not arg.class_no.isnumeric():
    logging.error("Argument --class_no must be numeric")
else:
    remove_class_data(int(arg.class_no))
