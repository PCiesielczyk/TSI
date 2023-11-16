import csv

from TSI.data_storage.file_loader import classes_csv_file_path


def csv_to_dict() -> dict:
    with open(classes_csv_file_path, 'r') as csvfile:
        class_dict = {}
        csv_reader = csv.reader(csvfile)

        for row in csv_reader:
            class_id, class_name = row
            class_dict[int(class_id)] = class_name

        return class_dict


class ClassMapper:
    def __init__(self):
        self.classes = csv_to_dict()

    def map_to_text(self, numeric_class):
        return self.classes.get(numeric_class, "Unknown class")
