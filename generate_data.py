from pathlib import Path


current_directory = Path.cwd()
print(f"Current directory: {current_directory}")

for path in sorted(current_directory.iterdir()):
	print(path.name)
