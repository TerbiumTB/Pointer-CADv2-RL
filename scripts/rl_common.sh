#!/usr/bin/env bash

rl_activate_conda() {
    local environment_name="$1"
    local conda_executable="${CONDA_EXE:-}"

    if [[ "${CONDA_DEFAULT_ENV:-}" == "$environment_name" ]]; then
        return 0
    fi

    if [[ -z "$conda_executable" ]]; then
        conda_executable="$(command -v conda || true)"
    fi
    if [[ -z "$conda_executable" ]]; then
        local candidate
        for candidate in \
            "$HOME/miniconda3/bin/conda" \
            "/opt/miniconda3/bin/conda" \
            "/root/miniconda3/bin/conda"; do
            if [[ -x "$candidate" ]]; then
                conda_executable="$candidate"
                break
            fi
        done
    fi
    if [[ -z "$conda_executable" || ! -x "$conda_executable" ]]; then
        echo "[ERROR] Conda executable was not found." >&2
        return 1
    fi

    local conda_base
    conda_base="$("$conda_executable" info --base)"
    source "$conda_base/etc/profile.d/conda.sh"
    conda activate "$environment_name"
}


rl_prepare_environment() {
    local repository_root="$1"
    local environment_name="$2"
    local hf_cache_dir="${3:-}"
    local proxy_script="${4:-}"

    cd "$repository_root"
    rl_activate_conda "$environment_name"
    export PYTHONUNBUFFERED=1

    if [[ -n "$hf_cache_dir" ]]; then
        mkdir -p "$hf_cache_dir"
        export HF_HOME="$hf_cache_dir"
    fi
    if [[ -n "$proxy_script" ]]; then
        if [[ ! -f "$proxy_script" ]]; then
            echo "[ERROR] Proxy script not found: $proxy_script" >&2
            return 1
        fi
        source "$proxy_script"
    fi
}


rl_require_file() {
    local path="$1"
    local description="$2"
    if [[ ! -f "$path" ]]; then
        echo "[ERROR] $description not found: $path" >&2
        return 1
    fi
}


rl_gpu_count() {
    local gpu_ids="$1"
    if [[ -z "$gpu_ids" ]]; then
        echo 0
        return
    fi
    local values="${gpu_ids//[^,]/}"
    echo $((${#values} + 1))
}


rl_timestamp() {
    date +"%Y%m%d_%H%M%S"
}
