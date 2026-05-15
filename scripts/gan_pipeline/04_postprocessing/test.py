import pathlib

# 1. Get the absolute path of the current script
# __file__ is a built-in variable pointing to the current script
script_path = pathlib.Path(__file__).resolve()

# 2. Get the Current Working Directory (where you ran the command from)
cwd = pathlib.Path().resolve()

# 3. Calculate the path from CWD to the file
path_to_file = script_path.relative_to(cwd)

print(f"CWD: {cwd}")
print(f"Script: {script_path}")
print(f"Path from CWD to file: {path_to_file}")