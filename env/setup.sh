#!/bin/bash                                                                                                                                                                                                                                         

call_path=$PWD
script_full_path=$(dirname "${BASH_SOURCE[0]}")

cd $script_full_path
 
export NUTRIG_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"
echo "Set var NUTRIG_ROOT="$NUTRIG_ROOT
echo "=============================="

export PYTHONPATH=$NUTRIG_ROOT:$PYTHONPATH
echo "add nutrig to PYTHONPATH"
echo "=============================="
