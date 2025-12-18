import torch
import pandas as pd
from zipfile import ZipFile
from io import BytesIO
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import os
from .transforms import get_val_transforms
from .models import XceptionResNet50


def predict_from_zip(zip_path, model_path, img_size=196, device='cuda', output_csv='predictions.csv'):
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = XceptionResNet50(num_classes=2).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_transform = get_val_transforms(img_size)
    TEST_IMAGES_ZIP_PATH = 'dataset/test_images'
    
    if not os.path.exists(zip_path):
        print(f"❌ Zip-archive {zip_path} not found")
        return
    
    print(f"✓ Found zip-archive: {zip_path}")
    
    with ZipFile(zip_path, 'r') as zip_file:
        all_files = zip_file.namelist()
        test_image_files = [
            f for f in all_files 
            if f.startswith(TEST_IMAGES_ZIP_PATH) and 
            (f.lower().endswith('.png') or f.lower().endswith('.jpg') or f.lower().endswith('.jpeg'))
        ]
        
        if len(test_image_files) == 0:
            print(f"❌ Images not found in {TEST_IMAGES_ZIP_PATH}")
            return
        
        print(f"✓ Found {len(test_image_files)} images in zip")
        test_image_files.sort()
        
        results = []
        batch_size = 32
        
        print(f"\nStart processing from zip...")
        with torch.no_grad():
            for i in tqdm(range(0, len(test_image_files), batch_size), desc="Processing from zip"):
                batch_files = test_image_files[i:i+batch_size]
                batch_images = []
                batch_ids = []
                
                for img_path_in_zip in batch_files:
                    try:
                        img_name = os.path.basename(img_path_in_zip)
                        img_id = Path(img_name).stem
                        batch_ids.append(img_id)
                        
                        with zip_file.open(img_path_in_zip) as f:
                            image = Image.open(BytesIO(f.read())).convert('RGB')
                            image_tensor = test_transform(image)
                            batch_images.append(image_tensor)
                    except Exception as e:
                        print(f"⚠️  Error loading {img_path_in_zip}: {e}")
                        continue
                
                if len(batch_images) == 0:
                    continue
                
                batch_tensor = torch.stack(batch_images).to(device)
                predictions = model(batch_tensor)
                pred_classes = predictions.argmax(dim=1).cpu().numpy()
                
                for img_id, pred_class in zip(batch_ids, pred_classes):
                    results.append({
                        'id': img_id,
                        'label': int(pred_class)
                    })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('id')
    results_df.to_csv(output_csv, index=False)
    
    print(f"\n{'='*60}")
    print(f"✅ Processing completed!")
    print(f"📊 Processed images: {len(results)}")
    print(f"📁 Results saved to: {output_csv}")
    print(f"{'='*60}")
    
    print(f"\n📋 First 10 rows of CSV:")
    print(results_df.head(10).to_string(index=False))
    
    check_df = pd.read_csv(output_csv)
    print(f"\n✓ CSV check:")
    print(f"  Columns: {list(check_df.columns)}")
    print(f"  Found column 'id': {'id' in check_df.columns}")
    print(f"  Rows in CSV: {len(check_df)}")
