delete_raw_files(){
  local folder="$1"
  if ls "$folder"/*.raw 1> /dev/null 2>&1; then
    rm "$folder"/*.raw
  fi
}
delete_raw_files "/home/nv/wyk/Data/light/input"
delete_raw_files "/home/nv/wyk/Data/light/output"