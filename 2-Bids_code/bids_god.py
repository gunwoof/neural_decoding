import os
import re
from tqdm import tqdm
import subprocess

# 원본 데이터와 출력 폴더 경로
source_dir = "/nas-tmp/research/03-Neural_decoding/1-raw_data/god"
output_dir = "/nas-tmp/research/03-Neural_decoding/3-bids"

# 피험자 목록 (예: 8명의 피험자)
subjects = [d for d in os.listdir(source_dir) if d.startswith("sub-")] # sub모음
os.makedirs(output_dir, exist_ok=True)

for i, sub in tqdm(enumerate(subjects,start=1), desc=f"Processing subject"):
    new_sub = f"sub-{i+12:02d}" # BIDS의 이름은 sub-<index>로 해야함
    sub_dir = os.path.join(output_dir, new_sub)  # BIDS 폴더: sub
    os.makedirs(sub_dir, exist_ok=True)

    subject_path = os.path.join(source_dir, sub)  # Raw 폴더: sub
    sessions = [d for d in os.listdir(subject_path) if d.startswith("ses-perception") and d[-2:].isdigit()] # Raw 폴더 ses모음

    # sessions 처리
    for i, ses in tqdm(enumerate(sessions,start=1), desc=f"Processing {new_sub}"):
        ses_dir = os.path.join(sub_dir, f"ses-{i:02d}")  # BIDS 폴더: sub-ses
        os.makedirs(ses_dir, exist_ok=True)
        
        session_path = os.path.join(subject_path, ses)  # Raw 폴더: sub-ses
        
        # anat 폴더 처리
        anat_dir = os.path.join(ses_dir, "anat")
        os.makedirs(anat_dir, exist_ok=True)
        source_anat_nii_path = os.path.join(subject_path, "ses-anatomy", "anat", f"{sub}_ses-anatomy_T1w.nii.gz") # Raw 폴더: sub-ses-anat
        source_anat_json_path = os.path.join(source_dir, "T1w.json") # Raw 폴더

        # T1w.nii.gz 파일 복사
        if os.path.exists(source_anat_nii_path):
            target_anat_nii_filename = f"{new_sub}_ses-{i:02d}_T1w.nii.gz"
            target_anat_nii_path = os.path.join(anat_dir, target_anat_nii_filename)
            subprocess.run(["rsync", "-avz",  "--ignore-existing", source_anat_nii_path, target_anat_nii_path]) # 이미 대상 경로에 존재하는 파일은 복사하지 않음.

        # T1w.json 파일 복사
        if os.path.exists(source_anat_json_path):
            target_anat_json_filename = f"{new_sub}_ses-{i:02d}_T1w.json"
            target_anat_json_path = os.path.join(anat_dir, target_anat_json_filename)
            subprocess.run(["rsync", "-avz",  "--ignore-existing", source_anat_json_path, target_anat_json_path]) # 이미 대상 경로에 존재하는 파일은 복사하지 않음.
        
        # dwi 폴더 처리
        dwi_dir = os.path.join(ses_dir, "dwi")
        os.makedirs(dwi_dir, exist_ok=True)
                

        # func 폴더 처리
        func_dir = os.path.join(ses_dir, "func") # BIDS 폴더: sub-ses-func
        os.makedirs(func_dir, exist_ok=True)
        func_path = os.path.join(session_path, "func") # Raw 폴더: sub-ses-func
        if os.path.exists(func_path):
            for func_file in tqdm(os.listdir(func_path), desc=f"Processing {new_sub} ses-{i:02d} func"):
                if "events" in func_file or "bold" in func_file:
                    func_source = os.path.join(func_path, func_file)
                    func_target = os.path.join(func_dir, func_file.replace(sub, new_sub).replace(f"{ses}", f"ses-{i:02d}").replace("task-perception", "task-image"))
                    subprocess.run(["rsync", "-avz",  "--ignore-existing", func_source, func_target]) # 이미 대상 경로에 존재하는 파일은 복사하지 않음.