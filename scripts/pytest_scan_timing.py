import os
import subprocess
import time

def main():
    test_dir = "tests"
    test_files = [f for f in os.listdir(test_dir) if f.startswith("test_") and f.endswith(".py")]
    results = []
    for test_file in sorted(test_files):
        path = os.path.join(test_dir, test_file)
        start = time.time()
        subprocess.run(
            ["pytest", "--collect-only", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        elapsed = time.time() - start
        results.append((elapsed, test_file))
        print(f"{test_file}: {elapsed:.3f} seconds")
    print("\nSlowest test files (scanning):")
    for elapsed, test_file in sorted(results, reverse=True):
        print(f"{test_file}: {elapsed:.3f} seconds")

if __name__ == "__main__":
    main()
