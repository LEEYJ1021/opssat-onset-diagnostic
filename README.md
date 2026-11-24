
## Running
export OPSSAT_BASE_DIR=/path/to/working/dir   # must contain paper2_outputs/, causal_pipeline_outputs/ (or data/derived copies)
export OPSSAT_OUT_DIR=$OPSSAT_BASE_DIR/revision_outputs
pip install -r env/requirements.lock.txt
python pipeline/05_window_power/rev11_injection.py --quick
