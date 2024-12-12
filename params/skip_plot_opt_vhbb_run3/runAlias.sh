#!/bin/bash

# Function to execute the transformed command for dask execution
sdrun() {
    local workflow="$1"
    local channel="$2"
    local process="$3"
    local timestamp="$4"
    
    # Construct the actual command
    local cfg_file="VHccPoCo/cfg_${workflow}_${channel}.py"
    local custom_opts="VHccPoCo/params/skip_plot_opt_vhbb_run3/custom_run_options_vhbb_test.yaml"
    local output_dir="output_${workflow}_${channel}_${process}_${timestamp}_dask"
    local log_file="dask_out_${workflow}_${channel}_${process}_${timestamp}.log"

    # Execute the command
    runner --cfg "$cfg_file" \
           --executor dask@lxplus \
           --custom-run-options "$custom_opts" \
           -o "$output_dir" 2>&1 | tee "$log_file" &
}

# Function to execute the transformed command for local execution
sdlocalrun() {
    local workflow="$1"
    local channel="$2"
    local process="$3"
    local timestamp="$4"
    local limit_factor="$5"
    
    # Construct the actual command
    local cfg_file="VHccPoCo/cfg_${workflow}_${channel}.py"
    local output_dir="output_${workflow}_${channel}_${process}_dev_local"
    local log_file="${output_dir}.log"

    # Execute the command
    runner --cfg "$cfg_file" \
           -o "$output_dir" \
           --executor futures \
           -s 20 \
           -lf "$limit_factor" \
           -lc "$limit_factor" 2>&1 | tee "$log_file" &
}

# Export the functions for use in the current shell session
export -f sdrun
export -f sdlocalrun

# Provide feedback to the user
echo "Environment setup complete. Use the commands with the format:"
echo "sdrun <workflow> <channel> <process> <timestamp>"
echo "example: sdrun VHbb ZLL DY 20241209"
echo "sdlocalrun <workflow> <channel> <process> <timestamp> <limit_factor>"
