from pathlib import Path

from traitlets import This


current_directory = Path.cwd()
print(f"Current directory: {current_directory}")

for path in sorted(current_directory.iterdir()):
	print(path.name)


# This is test update

# This is a test update to check if the changes are reflected in  the repository.