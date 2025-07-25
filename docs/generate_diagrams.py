#!/usr/bin/env python3
"""
Script to generate PNG diagrams from PlantUML files
Usage: uv run docs/generate_diagrams.py /path/to/plantuml.jar
"""

import os
import sys
import subprocess
import glob
from pathlib import Path

def main():
    # Check arguments
    if len(sys.argv) != 2:
        print("Usage: uv run docs/generate_diagrams.py /path/to/plantuml.jar")
        print("Example: uv run docs/generate_diagrams.py ~/tools/plantuml.jar")
        sys.exit(1)

    plantuml_jar = sys.argv[1]

    # Check if PlantUML jar exists
    if not os.path.isfile(plantuml_jar):
        print(f"Error: PlantUML jar file not found at: {plantuml_jar}")
        print("Please download PlantUML from: https://plantuml.com/download")
        sys.exit(1)

    # Check if Java is available
    try:
        subprocess.run(["java", "-version"],
                      capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Java is not installed or not in PATH")
        print("Please install Java to run PlantUML")
        sys.exit(1)

    # Set up directories
    script_dir = Path(__file__).parent.absolute()
    docs_dir = script_dir
    assets_dir = docs_dir / "assets"

    # Create assets directory if it doesn't exist
    assets_dir.mkdir(exist_ok=True)

    print("Generating diagrams from PlantUML files...")
    print(f"PlantUML jar: {plantuml_jar}")
    print(f"Docs directory: {docs_dir}")
    print(f"Output directory: {assets_dir}")
    print()

    # Find all .puml files in docs directory only (not subdirectories)
    puml_pattern = str(docs_dir / "*.puml")
    puml_files = glob.glob(puml_pattern)

    if not puml_files:
        print(f"No .puml files found in {docs_dir} directory")
        return

    print(f"Found {len(puml_files)} .puml file(s):")
    for f in puml_files:
        print(f"  - {os.path.basename(f)}")
    print()

    # Process each PlantUML file
    processed = 0
    total = len(puml_files)

    for puml_file in puml_files:
        print(f"Processing: {os.path.basename(puml_file)}")

        # Build PlantUML command
        cmd = [
            "java",
            "-DPLANTUML_LIMIT_SIZE=8192",
            "-jar", plantuml_jar,
            "-tpng",
            "-o", str(assets_dir),
            puml_file
        ]

        try:
            # Run PlantUML
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                base_name = Path(puml_file).stem
                output_file = assets_dir / f"{base_name}.png"
                print(f"  ✓ Generated: {output_file}")
                processed += 1
            else:
                print(f"  ✗ Failed to generate: {os.path.basename(puml_file)}")
                print(f"    Exit code: {result.returncode}")
                if result.stderr:
                    print(f"    Error: {result.stderr.strip()}")

        except Exception as e:
            print(f"  ✗ Error processing {os.path.basename(puml_file)}: {e}")

        print()

    # Summary
    print("Summary:")
    print(f"  Processed: {processed}/{total} files")
    print(f"  Output directory: {assets_dir}")

    # List generated files
    if processed > 0:
        print()
        print("Generated files:")
        png_files = list(assets_dir.glob("*.png"))
        if png_files:
            for png_file in sorted(png_files):
                file_size = png_file.stat().st_size
                print(f"  - {png_file.name} ({file_size:,} bytes)")
        else:
            print("  No PNG files found in assets directory")

    print()
    print("Done!")

if __name__ == "__main__":
    main()
