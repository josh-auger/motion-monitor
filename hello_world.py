import sys

def hello_world():
    print("Hello, world!")

def print_filepath():
    if len(sys.argv) < 2:
        print("Usage: python3 hello_world.py <logfile_path>")
        sys.exit(1)

    logfile_path = sys.argv[1]
    print("Logfile path:", logfile_path)

if __name__ == "__main__":
    hello_world()
    print_filepath()