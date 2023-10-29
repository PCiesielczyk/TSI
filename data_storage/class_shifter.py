import os
import logging
import argparse
import csv

project_dir = os.path.dirname(os.getcwd())
archive_dir = os.path.join(project_dir, 'archive')
meta_csv_path = os.path.join(archive_dir, 'Meta.csv')
test_csv_path = os.path.join(archive_dir, 'Test.csv')
train_csv_path = os.path.join(archive_dir, 'Train.csv')

logging.basicConfig(level=logging.DEBUG)


def append_missing_meta(rows: list, out_class: int):
    for class_no in range(out_class):
        if not any(row[0] == str(class_no) for row in rows):
            new_row = ['Meta/' + str(class_no) + '.png', str(class_no)]
            rows.append(new_row)


def update_meta(in_class: int, out_class: int):
    with open(meta_csv_path, 'r') as meta_csv:
        reader = csv.reader(meta_csv)
        rows = list(reader)
        new_rows = []

        for row in rows:
            if row[1] == str(in_class):
                row[0] = 'Meta/' + str(out_class) + '.png'
                row[1] = out_class
            new_rows.append(row)

        append_missing_meta(new_rows, out_class)

    with open(meta_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(new_rows)
        logging.info('Processed Meta.csv file')


def update_test(in_class: int, out_class: int):
    with open(test_csv_path, 'r') as test_csv:
        reader = csv.reader(test_csv)
        rows = list(reader)
        new_rows = []

        for row in rows:
            if row[6] == str(in_class):
                row[6] = out_class
            new_rows.append(row)

    with open(test_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(new_rows)
        logging.info('Processed Test.csv file')


def update_train(in_class: int, out_class: int):
    with open(train_csv_path, 'r') as train_csv:
        reader = csv.reader(train_csv)
        rows = list(reader)
        new_rows = []

        for row in rows:
            if row[6] == str(in_class):
                row[6] = out_class
                image_path_segments = row[7].split('/')
                new_path = image_path_segments[0] + '/' + str(out_class) + '/' + image_path_segments[2]
                row[7] = new_path
            new_rows.append(row)

    with open(train_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(new_rows)
        logging.info('Processed Train.csv file')


def shift_class(in_class: int, out_class: int):
    update_meta(in_class, out_class)
    update_test(in_class, out_class)
    update_train(in_class, out_class)


parser = argparse.ArgumentParser(description="Rename class name from Meta, Train and test csv.")
parser.add_argument("--in_class", help="No. of class to be renamed")
parser.add_argument("--out_class", help="No. of class after rename")
arg = parser.parse_args()

if arg.in_class is None or arg.out_class is None:
    logging.error("Argument --in_class or --out_class not specified")
elif not arg.in_class.isnumeric() or not arg.out_class.isnumeric():
    logging.error("Argument --in_class and --out_class must be numeric")
else:
    shift_class(int(arg.in_class), int(arg.out_class))
