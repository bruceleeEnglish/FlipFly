import re
import os
import pandas as pd  # pandas


def truncate(n):
    """Truncate to two decimal places."""
    return int(n * 100) / 100


def extract_numbers_from_file(file_path):
    # Regular expression for the target line format
    pattern = r'ef(\d+): search time = (\d+\.\d+) \[ms\]  recall = (\d+\.\d+) per query'

    No_ = []
    QPS_ = []
    recalls_ = []

    all_time = 0

    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                # Extract the three numbers and convert them to int and float
                ef = int(match.group(1))
                search_time = float(match.group(2))
                all_time += search_time
                recall = float(match.group(3))

                No_.append(ef)
                QPS_.append(truncate(1000 / search_time * 100))  # truncated to two decimal places
                recalls_.append(str(truncate(recall * 100)) + "%")  # truncated to two decimal places

    print("all_time:", all_time)
    return No_, QPS_, recalls_


def print_list(lst):
    for l in lst:
        print(l, end='\t')
    print()


def list_all_files(directory):
    all_data = []  # Collect the data of all files

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            print(file_path)
            No, QPS, recalls = extract_numbers_from_file(file_path)
            print_list(No)
            print_list(QPS)
            print_list(recalls)
            print("-------------------------------------")


# Example call
directory = 'D://C++ plus//DCOs//src//script//log_ood'  # directory path
list_all_files(directory)
