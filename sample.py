# create_sample_nsl_kdd.py

sample_lines = [
    # 41 features + attack_type + difficulty, comma-separated, no header
    "0,tcp,http,SF,181,5450,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,9,9,0.00,0.00,0.00,0.00,1.00,0.00,0.00,9,9,1.00,0.00,0.11,0.00,0.00,0.00,0.00,0.00,normal,21",
    "0,tcp,http,SF,239,486,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,19,19,0.00,0.00,0.00,0.00,1.00,0.00,0.00,19,19,1.00,0.00,0.05,0.00,0.00,0.00,0.00,0.00,normal,21",
    "0,tcp,ftp_data,SF,235,1337,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,29,15,0.00,0.00,0.00,0.00,0.52,0.48,0.00,255,15,0.06,0.03,0.01,0.00,0.00,0.00,0.00,0.00,back,21",
]

output_file = "sample_nsl_kdd.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for line in sample_lines:
        f.write(line + "\n")

print(f"Created {output_file} with {len(sample_lines)} NSL-KDD-style rows.")
