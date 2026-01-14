import boto3
from botocore import UNSIGNED
from botocore.config import Config

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
BUCKET_NAME = 'noaa-goes16'
PREFIX = 'ABI-L2-CMIPF/2025/'

print(f"Checking days in {PREFIX}...")
try:
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=PREFIX, Delimiter='/')
    days = response.get('CommonPrefixes', [])
    day_prefixes = [p['Prefix'] for p in days]
    day_numbers = sorted([int(p.split('/')[-2]) for p in day_prefixes])
    
    if day_numbers:
        print(f"First 5 days: {day_numbers[:5]}")
        print(f"Last 5 days: {day_numbers[-5:]}")
    else:
        print("No days found in 2025.")
        
except Exception as e:
    print(f"Error: {e}")
