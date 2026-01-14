import boto3
from botocore import UNSIGNED
from botocore.config import Config
import json

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
BUCKET_NAME = 'noaa-goes16'
PREFIX = 'ABI-L2-CMIPF/'

print(f"Checking top level prefixes under {PREFIX} in {BUCKET_NAME}...")
try:
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=PREFIX, Delimiter='/')
    prefixes = response.get('CommonPrefixes', [])
    years = [p['Prefix'].split('/')[-2] for p in prefixes]
    print(f"Available Years: {years}")
    
    if years:
        last_year = sorted(years)[-1]
        print(f"Checking months for latest year ({last_year})...")
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=f"{PREFIX}{last_year}/", Delimiter='/')
        days = response.get('CommonPrefixes', []) # acts as days because format is year/day_of_year
        # Actually structure is year/day_of_year
        # Let's see some days
        day_prefixes = [p['Prefix'] for p in days[:5]] # first 5
        print(f"Sample days in {last_year}: {day_prefixes}")
        
except Exception as e:
    print(f"Error: {e}")
