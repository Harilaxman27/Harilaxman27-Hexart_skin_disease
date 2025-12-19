
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

# ==========================================
# PART 1: DATA ORGANIZATION
# ==========================================
def setup_dataset():
    print("🔄 Starting Data Organization...")
    
    # Paths (Adjust these if running locally)
    repo_name = "Harilaxman27-Hexart_skin_disease"
    source_dir = f"../dataset"  # Assuming script is in YOLOv8 folder
    csv_path = f"{source_dir}/metadata.csv"
    output_dir = "../YOLO_Ready_Dataset"

    if not os.path.exists(csv_path):
        print(f"❌ Error: Could not find {csv_path}")
        return

    # Read Metadata
    df = pd.read_csv(csv_path)

    # Split: 80% Train, 20% Val
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df['dx'], random_state=42
    )

    # Helper function to move images
    def move_images(dataframe, split_name):
        print(f"   Processing {split_name} data...")
        for index, row in dataframe.iterrows():
            disease = row['dx']
            img_name = row['image_id'] + ".jpg"
            
            src = os.path.join(source_dir, img_name)
            dest_folder = os.path.join(output_dir, split_name, disease)
            os.makedirs(dest_folder, exist_ok=True)
            
            if os.path.exists(src):
                shutil.copy(src, os.path.join(dest_folder, img_name))

    # Execute Move
    move_images(train_df, "train")
    move_images(val_df, "val")
    print("✅ Data Organization Complete!")

# ==========================================
# PART 2: MODEL TRAINING
# ==========================================
def train_model():
    print("🚀 Starting YOLOv8 Training...")
    
    # Load the Small model (Smarter than Nano)
    model = YOLO('yolov8s-cls.pt')

    # Train
    results = model.train(
        data='../YOLO_Ready_Dataset', 
        epochs=50, 
        imgsz=224, 
        batch=16,
        patience=10,
        name='skin_disease_yolov8'
    )
    print("✅ Training Complete!")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Prepare Data
    setup_dataset()
    
    # 2. Train
    train_model()
