#!/usr/bin/env bash
DEFAULT_PROXY="http://10.10.10.102:7999"
export http_proxy="$DEFAULT_PROXY"
export https_proxy="$DEFAULT_PROXY"
echo "Proxy has been set:"
echo "http_proxy=$http_proxy"
echo "https_proxy=$https_proxy"
