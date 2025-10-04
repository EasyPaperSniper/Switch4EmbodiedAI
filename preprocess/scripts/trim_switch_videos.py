import subprocess

# Dictionary mapping filename -> (start_time, end_time)
# Times are in "HH:MM:SS" format
cuts = {
    "Baby_Shark.mkv": ("00:00:16", "00:01:36"),
    "bad_guy.mkv": ("00:00:00", "00:03:09"),
    "Heart_Of_Glass.mkv": ("00:00:08", "00:03:44"),
    "Old_Town_Road.mkv": ("00:00:00", "00:02:41"),
    "Padam_Padam.mkv": ("00:00:25", "00:02:54"),
    "Soy_Yo.mkv": ("00:00:17", "00:02:49"),
    "Unstoppable.mkv": ("00:00:12", "00:03:36"),
}

infile_dir = "/home/jkim3662/Projects/Switch4EmbodiedAI/data/switch_videos/raw"
outfile_dir = "/home/jkim3662/Projects/Switch4EmbodiedAI/data/switch_videos/cut"

for infile, (start, end) in cuts.items():
    # output file name -> same base, but with _cut.mp4 suffix
    infile_path = f"{infile_dir}/{infile}"
    outfile = infile.rsplit(".", 1)[0] + "_cut.mp4"
    outfile_path = f"{outfile_dir}/{outfile}"
    print(f"Cutting {infile} -> {outfile} ({start} to {end})")

    # Run ffmpeg with re-encoding for web compatibility
    cmd = [
        "ffmpeg", "-i", infile_path,
        "-ss", start, "-to", end,
        "-c:v", "libx264",           # H.264 video codec
        "-preset", "medium",         # Balance between speed and compression
        "-crf", "23",                # Quality (lower = better, 18-28 is good range)
        "-c:a", "aac",               # AAC audio codec
        "-b:a", "128k",              # Audio bitrate
        "-movflags", "+faststart",   # Enable progressive streaming for web
        outfile_path
    ]
    subprocess.run(cmd, check=True)

print("All cuts completed!")
