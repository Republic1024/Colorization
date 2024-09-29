:: 获取当前文件夹名
for %%I in ("%%cd%%") do set "folder_name=Colorization"

:: 使用当前文件夹名创建一个新的 GitHub 仓库
gh repo create "Colorization" --public

:: 初始化一个新的 Git 仓库
git init

:: 将远程仓库添加为 origin
git remote add origin https://github.com/Republic1024/Colorization.git

:: 从远程仓库的 main 分支拉取最新代码
git pull origin main

:: 移除未跟踪的文件，例如 .DS_Store（Mac 系统生成的隐藏文件）
del /F /Q .DS_Store

:: 将所有更改添加到暂存区，排除 .pth 文件
git add .

:: 提交暂存区的更改并添加提交信息
git commit -m "Initial commit or merge or rebase remote changes"

:: 将本地的 main 分支变基到远程 main 分支的最新提交
git branch -M main

:: 将本地 main 分支的更改推送到远程 main 分支
git push -u origin main