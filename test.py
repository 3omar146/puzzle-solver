from src.pipeline import process_dataset, print_accuracy_table

process_dataset("data/raw/Gravity Falls/puzzle_2x2")
process_dataset("data/raw/Gravity Falls/puzzle_4x4")
process_dataset("data/raw/Gravity Falls/puzzle_8x8")
print_accuracy_table()