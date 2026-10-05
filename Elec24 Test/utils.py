
from pathlib import Path
import pandas as pd


# This function loads all CSV chunk files from a specified part folder and yields each chunk as a DataFrame.
# This function is basically to look at the schema to determine the differences. Also for checking warnings for each chunk for different types
def load_chunks_by_part(part_folder: Path):
    chunk_files = sorted(part_folder.glob("*.csv.gz"))
    if not chunk_files:
        print(f"No chunk files found in {part_folder.name}")
        return

    for chunk_file in chunk_files:
        print(f"  Loading chunk: {chunk_file.name}")
        df_chunk = pd.read_csv(chunk_file)
        yield chunk_file.name, df_chunk


# This function loads a specified number of CSV chunks files. This will be used for testing purposes to avoid loading the entire dataset.
def load_sample_chunks(part_dir: Path, num_chunks: int = 1, as_str: bool = True) -> pd.DataFrame:
    part_path = Path(part_dir)
    chunk_files = sorted(part_path.glob("*.csv.gz"))[:num_chunks]

    if not chunk_files:
        print(f"No '.csv.gz' chunk files found in {part_path}")
        return pd.DataFrame()

    read_kwargs = {"dtype": str} if as_str else {}
    
    dfs = []
    for f in chunk_files:
        print(f"Loading chunk: {f.name}")
        dfs.append(pd.read_csv(f, **read_kwargs))

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Successfully combined {len(chunk_files)} chunk(s). Total rows: {len(combined_df)}")
    
    return combined_df


# Function to look into the schema of each part that is different from the others
# Helps in contextualizing the differences in the schema of each part
def summarize_part_schemas(dataset_path: Path, part_numbers: list[int]) -> pd.DataFrame:
    schema_records = []

    for part_num in part_numbers:
        part_dir = dataset_path / f"part_{part_num}"
        chunk_files = sorted(part_dir.glob("*.csv.gz"))
        
        if not chunk_files:
            print(f"Warning: No chunks found in part_{part_num}")
            continue
            
        # Read only the header and first row to quickly infer schema
        first_chunk = chunk_files[0]
        df_sample = pd.read_csv(first_chunk, nrows=1)
        
        schema_records.append({
            "Part": f"part_{part_num}",
            "First Chunk": first_chunk.name,
            "Total Chunks": len(chunk_files),
            "Column Count": len(df_sample.columns),
            "Columns": list(df_sample.columns),
            "Data Types": dict(df_sample.dtypes.astype(str))
        })

    return pd.DataFrame(schema_records)