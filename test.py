from src.pipeline import process_dataset
from src.utils import delete_images_in_folder
delete_images_in_folder("data/enhanced")
process_dataset("data/raw/Gravity Falls/puzzle_2x2")