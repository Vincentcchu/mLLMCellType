# mLLMCellType Agent

Multi-modal Large Language Model approach for cell type classification.

## Overview

mLLMCellType uses multi-modal LLM techniques to perform cell type annotation on single-cell RNA-seq data.

## Prerequisites

- Python 3.8+
- API access to language models
- Required Python packages

## Setup

1. **Navigate to agent directory**:
   ```bash
   cd agents/mllmcelltype
   ```

2. **Install dependencies**:
   ```bash
   pip install scanpy anndata numpy pandas
   # Additional packages as needed
   ```

3. **Configure API access**:
   - Set environment variables or configuration for LLM API access
   - Check code for specific configuration requirements

## Input Data

- **Expected format**: Standard AnnData h5ad file
- **Expected location**: `../../data/dataset_restricted.h5ad`
- **Required fields**: Gene expression matrix

### Data Preparation (if needed):

```bash
cd ../../preprocessing/agent_specific/mllmcelltype/
jupyter notebook data_preprocessing.ipynb
```

## Running

### Main entrypoint:

```bash
cd agents/mllmcelltype
python run.py
```

### Alternative run modes:

- `run_log.py` - Run with logging
- `run_top.py` - Run with top-k gene selection

### Example:

```bash
# Standard run
python run.py

# With logging
python run_log.py

# Top genes mode
python run_top.py
```

## Output

Results are saved to:
- **Default location**: `../../outputs/mllmcelltype/`
- **Output format**: Cell type predictions
- **Logs**: `mllm_run.log` (when using `run_log.py`)

## Configuration

- Edit paths and parameters directly in the run scripts
- Configure API endpoints and credentials as needed
- Adjust gene selection and preprocessing parameters

## Notes

- Multiple run scripts provide different execution modes
- Check individual scripts for specific configuration options
- This agent may be a git submodule pointing to an external repository

## Troubleshooting

**Issue**: API connection errors
- **Solution**: Verify API credentials and network connectivity

**Issue**: Data file not found
- **Solution**: Ensure data is at `../../data/dataset_restricted.h5ad`

**Issue**: Memory errors
- **Solution**: Reduce batch size or use subset of data for testing

## Additional Resources

- Data preprocessing: [data_preprocessing.ipynb](data_preprocessing.ipynb)
- Run logs: Check `mllm_run.log` for execution details
- Output directory: `../../outputs/mllmcelltype/`
