import os
import sys
import tempfile
import dashscope

# Load API Key from config
try:
    sys.path.append(os.getcwd())
    import config
    # Use dashcope key (typo in config 'dashcope' is known)
    api_key = getattr(config, 'dashcope', None)
    if not api_key:
        # Fallback to modelscope key if dashscope missing
        api_key = getattr(config, 'modelscope', None)
        print("Using modelscope key as fallback")
    
    if not api_key:
        print("Error: No API Key found.")
        sys.exit(1)
        
    dashscope.api_key = api_key
    print(f"API Key loaded: {api_key[:6]}...")
except Exception as e:
    print(f"Config load error: {e}")
    sys.exit(1)

# Create dummy PNG
with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
    # Valid PNG signature
    tmp.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82')
    tmp_path = tmp.name

print(f"Created temp file: {tmp_path}")

try:
    # 1. Test Files.upload with path (Expect UTF-8 error if previous observation correct)
    print("\nTest 1: dashscope.files.Files.upload(path)")
    try:
        from dashscope.files import Files
        res = Files.upload(tmp_path, description="test_upload_path")
        print(f"SUCCESS: {res}")
    except Exception as e:
        print(f"FAILED: {e}")

    # 3. Test FileUploader if exists
    print("\nTest 3: Checking for FileUploader or other classes")
    if hasattr(dashscope, 'FileUploader'):
        print("Found dashscope.FileUploader")
    else:
        print("dashscope.FileUploader NOT found")
        # List attributes to spy
        # print(dir(dashscope))

    # 4. Check dashscope.files structure
    import dashscope.files
    print(f"\ndashscope.files attributes: {dir(dashscope.files)}")

    # 5. Try using requests manually
    print("\nTest 4: Manual Request Attempt")
    import requests
    url = "https://dashscope.aliyuncs.com/api/v1/files"
    
    headers = {"Authorization": f"Bearer {api_key}"}
    
    # Try different purposes or none
    # purpose: 'file-upload' is a guess, usually strictly validated
    
    print(f"POST {url} with file...")
    try:
        with open(tmp_path, 'rb') as f:
            files = {'file': ('test.png', f, 'image/png')}
            data = {'purpose': 'general'} # Guessing purpose
            
            resp = requests.post(url, headers=headers, files=files, data=data) 
            print(f"Response status: {resp.status_code}")
            print(f"Response text: {resp.text}")
            
            if resp.status_code == 200:
                rj = resp.json()
                file_id = rj['data']['uploaded_files'][0]['file_id']
                print(f"File ID: {file_id}")
                
                # Check detail
                detail_url = f"{url}/{file_id}"
                print(f"GET {detail_url}...")
                resp_d = requests.get(detail_url, headers=headers)
                print(f"Detail Response: {resp_d.text}")
            
            elif resp.status_code != 200:
                 # Try without purpose
                 f.seek(0)
                 print("Retrying without purpose...")
                 files = {'file': ('test.png', f, 'image/png')}
                 resp = requests.post(url, headers=headers, files=files)
                 print(f"Response status: {resp.status_code}")
                 print(f"Response text: {resp.text}")
                 if resp.status_code == 200:
                     rj = resp.json()
                     file_id = rj['data']['uploaded_files'][0]['file_id']
                     print(f"File ID: {file_id}")
                     detail_url = f"{url}/{file_id}"
                     resp_d = requests.get(detail_url, headers=headers)
                     print(f"Detail Response: {resp_d.text}")

    except Exception as e:
        print(f"Manual upload failed: {e}")

finally:
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
