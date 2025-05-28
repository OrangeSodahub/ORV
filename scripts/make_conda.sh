# 1,执行此bash脚本
# 2,找聪哥修改HOME路径(需要root权限)
# 3,所有vscode页面均需要重新打开
# 4,所有终端均需要重新打开
# 5,输入命令 echo $PATH | grep -o '[^:]*miniconda[^:]*' ，如果出现类似，则conda base代表转换成功：
# /share/project/cwm/yuhao.duan/miniconda3/bin
# /share/project/cwm/yuhao.duan/miniconda3/condabin
# 如果出现类似，则代表conda base转换失败：
# /share/project/algorithm/yuhao.duan/miniconda3/bin
# /share/project/algorithm/yuhao.duan/miniconda3/condabin
# 6,conda activate your_env 后 输入 echo $PATH | grep -o '[^:]*miniconda[^:]*' ，若conda your_env转换成功：
# /share/project/cwm/yuhao.duan/miniconda3/envs/your_env/bin
# /share/project/cwm/yuhao.duan/miniconda3/condabin
# 若conda your_env失败：
# /share/project/algorithm/yuhao.duan/miniconda3/envs/your_env/bin
# /share/project/algorithm/yuhao.duan/miniconda3/condabin


NEW_DIR="/root/.conda"
OLD_DIR="/baai-cwm-1/baai_cwm_ml/cwm/xiuyu.yang/.conda"


# 函数：递归遍历目录并替换路径
update_paths_in_files() {
    local target_dir="$1"
    echo "Processing directory: $target_dir"
    
    # 遍历目标目录下的所有文件
    find "$target_dir" -type f | while read -r file; do
        # 检查文件内容是否包含旧路径
        if grep -q "$OLD_DIR" "$file"; then
            # 替换文件中所有的旧路径为新路径
            sed -i "s|$OLD_DIR|$NEW_DIR|g" "$file"
            echo "Updated $file"
        fi
    done
}

echo "Updating files in conda base directory..."
for file in "$NEW_DIR/bin/"*; do
    if head -n1 "$file" | grep -q "$OLD_DIR"; then
        sed -i "1 s|$OLD_DIR|$NEW_DIR|" "$file"
        echo "Updated $file"
    fi
done

# 遍历 envs 目录下的所有子目录
echo "Updating files in envs directory..."
for env_dir in "$NEW_DIR/envs"/*/; do
    # 检查是否是目录
    if [ -d "$env_dir" ]; then
        # 处理每个子目录下的 bin 文件夹中的文件
        for file in "$env_dir/bin/"*; do
            # 检查是否是文件
            if [ -f "$file" ]; then
                # 检查文件的第一行是否包含旧路径
                if head -n1 "$file" | grep -q "$OLD_DIR"; then
                    # 替换文件第一行的旧路径为新路径
                    sed -i "1 s|$OLD_DIR|$NEW_DIR|" "$file"
                    echo "Updated $file"
                fi
            fi
        done
    fi
done

# 更新 shell 目录中的文件
echo "Updating files in shell directory..."
update_paths_in_files "$NEW_DIR/shell/condabin"

# 更新 etc 目录中的文件
echo "Updating files in etc directory..."
update_paths_in_files "$NEW_DIR/etc/fish"
update_paths_in_files "$NEW_DIR/etc/profile.d"