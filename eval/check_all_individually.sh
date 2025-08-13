# repeatedly call check_all.py for every file in the args
#!/bin/bash


for file in "$@"; do
    echo "Checking $file"
    python3 eval/check_all.py "$file"
    if [ $? -ne 0 ]; then
        echo "Error checking $file"
    fi
done
