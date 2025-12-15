from src.pipeline import process_dataset, print_accuracy_table

auto_size_detection = False

process_dataset("data/raw/Gravity Falls/puzzle_2x2", auto_size_detection)
process_dataset("data/raw/Gravity Falls/puzzle_4x4", auto_size_detection)
process_dataset("data/raw/Gravity Falls/puzzle_8x8", auto_size_detection)

if auto_size_detection:
    print_accuracy_table()