import os
import pandas as pd
import numpy as np


def reconstruct_ysf_from_talys(
    talys_root_dir,
    output_csv,
):
    """
    Read TALYS input files from sample_* directories and
    reconstruct a DataFrame formatted like Bayes_ysf_post_params.py
    """

    rows = []

    # Loop through sample directories
    for subdir in sorted(os.listdir(talys_root_dir)):

        sample_path = os.path.join(talys_root_dir, subdir)
        if not os.path.isdir(sample_path):
            continue

        talys_file = os.path.join(sample_path, "input")
        if not os.path.exists(talys_file):
            continue

        # Initialize storage for parameters
        params = {
            "wtable": None,
            "Etable": None,
            "ftable": None,
            "upbendc": None,
            "upbende": None,
            "upbendf": None,
            "beta2": None,
        }

        # Read file
        with open(talys_file, "r") as f:
            for line in f:
                
                parts = line.strip().split()

                if len(parts) == 5:
                    
                    name, ZZ, AA, value, XL = parts
                    if name in params:
                        params[name] = float(value)
                elif len(parts) == 4 and parts[0] == "beta2":
                    
                    name, ZZ, AA, value = parts
                    params[name] = float(value)
        # Ensure all parameters were found
        if all(v is not None for v in params.values()):
            rows.append(params)
        else:
            print(f"Warning: Missing parameters in {talys_file}")

    # Convert to DataFrame
    df = pd.DataFrame(rows)

    # Add placeholder LogLikelihood column
    df["LogLikelihood"] = np.nan

    # Reorder columns exactly as original format
    df = df[
        [
            "wtable",
            "Etable",
            "ftable",
            "upbendc",
            "upbende",
            "upbendf",
            "beta2",
            "LogLikelihood",
        ]
    ]

    # Write output
    df.to_csv(output_csv, index=False)

    print(f"Reconstructed ySF posterior file written to {output_csv}")
    print(f"Recovered {len(df)} samples.")

def main():
    reconstruct_ysf_from_talys(talys_root_dir="/evtdata/spyrou-sim/jordan/Cu59_py_astro-y/",
                                output_csv="posterior_files/Reconstructed_Bayes_ysf_output.csv"
                                )
if __name__ == "__main__":
    main()