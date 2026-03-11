import pandas as pd
import os
from sklearn.model_selection import train_test_split

def main():
    metadata_path = "datasets/kaggle_dfdc/metadata.csv"
    images_dir = "datasets/kaggle_dfdc/faces_224"
    out_dir = "datasets/manifests"
    
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading Kaggle DFDC metadata...")
    df = pd.read_csv(metadata_path)
    
    print("Preparing manifest...")
    records = []
    
    # Fast check of existing files since there are ~100k
    existing_files = set(os.listdir(images_dir))
    
    for _, row in df.iterrows():
        img_name = row['videoname'].replace('.mp4', '.jpg')
        if img_name in existing_files:
            label = 1 if row['label'].upper() == 'FAKE' else 0
            path = os.path.join(images_dir, img_name).replace('\\', '/')
            records.append({
                'path': path,
                'label': label,
                'source': 'kaggle_dfdc'
            })
            
    final_df = pd.DataFrame(records)
    print(f"Total valid images compiled: {len(final_df)}")
    
    if len(final_df) == 0:
        print("Error: No images found! Make sure the path is exactly E:\\Deepfake\\AntiGravity-Deepfake-Detection\\datasets\\kaggle_dfdc")
        return

    # Split: 80% Train, 10% Val, 10% Test
    train_df, temp_df = train_test_split(final_df, test_size=0.2, random_state=42, stratify=final_df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    
    train_df.to_csv(f"{out_dir}/train.csv", index=False)
    val_df.to_csv(f"{out_dir}/val.csv", index=False)
    test_df.to_csv(f"{out_dir}/test.csv", index=False)
    
    print(f"Successfully generated datasets:")
    print(f" - Train: {len(train_df)} images")
    print(f" - Val:   {len(val_df)} images")
    print(f" - Test:  {len(test_df)} images")
    print(f"Manifests saved to {out_dir}/")
    
if __name__ == '__main__':
    main()
