package main

import (
	"archive/zip"
	"bufio"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/hidevopsio/hiboot/pkg/app"
	"github.com/hidevopsio/hiboot/pkg/app/cli"
)

func copyDir(src, dst string) error {
	srcInfo, err := os.Stat(src)
	if err != nil {
		return err
	}

	if err = os.MkdirAll(dst, srcInfo.Mode()); err != nil {
		return err
	}

	entries, err := os.ReadDir(src)
	if err != nil {
		return err
	}

	for _, entry := range entries {
		srcPath := filepath.Join(src, entry.Name())
		dstPath := filepath.Join(dst, entry.Name())

		if entry.IsDir() {
			if err = copyDir(srcPath, dstPath); err != nil {
				return err
			}
		} else {
			if err = copyFile(srcPath, dstPath); err != nil {
				return err
			}
		}
	}
	return nil
}

func copyFile(src, dst string) error {
	srcFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer srcFile.Close()

	dstFile, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer dstFile.Close()

	if _, err = io.Copy(dstFile, srcFile); err != nil {
		return err
	}

	srcInfo, err := os.Stat(src)
	if err != nil {
		return err
	}

	return os.Chmod(dst, srcInfo.Mode())
}

func printColorful(text string, colorCode string) {
	fmt.Printf("\033[%sm%s\033[0m\n", colorCode, text)
}

func progressBar(current, total int, description string) {
	width := 50
	percent := float64(current) / float64(total)
	filled := int(float64(width) * percent)
	bar := strings.Repeat("=", filled) + strings.Repeat(" ", width-filled)
	fmt.Printf("\r[%s] %3.0f%% | 步骤 %d/%d: %s", bar, percent*100, current, total, description)
	if current == total {
		fmt.Println()
	}
}

func confirmDeployment() bool {
	printColorful("警告：你即将开始部署。请输入 'y' 确认部署，或任意其他键退出。", "91")
	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("请输入确认字符 (y/n): ")
		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)
		if strings.ToLower(input) == "y" {
			printColorful("部署确认。", "32")
			return true
		} else if strings.ToLower(input) == "n" {
			printColorful("部署已取消。", "91")
			return false
		} else {
			printColorful("无效输入，请输入 'y' 或 'n'。", "91")
		}
	}
}

func runCommand(command []string, description string, colorCode string) bool {
	printColorful("PFAI自动化脚本01即将启动："+description+" ("+strings.Join(command, " ")+")", colorCode)

	if runtime.GOOS == "linux" {
		// 处理Linux平台特定的命令
		if command[0] == "pip" {
			command[0] = "pip3"
		}
	}

	cmd := exec.Command(command[0], command[1:]...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err := cmd.Run()
	if err != nil {
		printColorful("命令执行失败: "+err.Error(), "91")
		return false
	}
	return true
}

func cloneNltkData() error {
	printColorful("正在克隆nltk_data项目......", "34")

	// 克隆nltk_data仓库
	//if !runCommand([]string{"git", "clone", "https://github.com/nltk/nltk_data.git"}, "克隆nltk_data仓库", "34") {
	//	return fmt.Errorf("克隆nltk_data仓库失败")
	//}

	// 设置目标目录
	targetDir := "/root/nltk_data"
	if err := os.MkdirAll(targetDir, 0755); err != nil {
		return fmt.Errorf("创建目标目录失败: %v", err)
	}

	// 下载必要的NLTK数据文件
	//filesToDownload := []string{"tokenizers/punkt.zip", "tokenizers/punkt_tab.zip"}
	//baseURL := "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/"

	//for _, file := range filesToDownload {
	//	url := baseURL + file
	//	filePath := filepath.Join(targetDir, file)
	//
	//	printColorful(fmt.Sprintf("正在下载: %s", url), "34")
	//	if !runCommand([]string{"curl", "-L", url, "-o", filePath}, "下载NLTK数据文件", "34") {
	//		return fmt.Errorf("下载文件失败: %s", file)
	//	}
	//
	//	// 解压下载的文件
	//	if strings.HasSuffix(file, ".zip") {
	//		printColorful(fmt.Sprintf("正在解压: %s", filePath), "34")
	//		if err := unzip(filePath, filepath.Dir(filePath)); err != nil {
	//			printColorful(fmt.Sprintf("解压文件失败: %v", err), "91")
	//		}
	//	}
	//}

	// 清理临时目录
	if err := os.RemoveAll("nltk_data"); err != nil {
		printColorful(fmt.Sprintf("清理临时目录失败: %v", err), "91")
	}

	// 解压zip文件
	localZipPath := filepath.Join(filepath.Dir(os.Args[0]), ".", "punkt_tab.zip")
	if _, err := os.Stat(localZipPath); err != nil {
		printColorful(fmt.Sprintf("本地zip文件不存在: %s", localZipPath), "33")
		return nil
	}

	extractDir := filepath.Join(targetDir, "tokenizers")
	if err := os.MkdirAll(extractDir, 0755); err != nil {
		return fmt.Errorf("创建解压目录失败: %v", err)
	}

	printColorful(fmt.Sprintf("正在解压: %s", localZipPath), "34")
	if err := unzip(localZipPath, extractDir); err != nil {
		printColorful(fmt.Sprintf("解压文件失败: %v", err), "91")
	}

	return nil
}

func unzip(src, dest string) error {
	r, err := zip.OpenReader(src)
	if err != nil {
		return err
	}
	defer r.Close()

	for _, f := range r.File {
		path := filepath.Join(dest, f.Name)

		if f.FileInfo().IsDir() {
			os.MkdirAll(path, f.Mode())
			continue
		}

		if err := os.MkdirAll(filepath.Dir(path), f.Mode()); err != nil {
			return err
		}

		outFile, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, f.Mode())
		if err != nil {
			return err
		}

		rc, err := f.Open()
		if err != nil {
			outFile.Close()
			return err
		}

		_, err = io.Copy(outFile, rc)
		outFile.Close()
		rc.Close()
		if err != nil {
			return err
		}
	}

	return nil
}

type rootCommand struct {
	cli.RootCommand
	to string
}

func newRootCommand() *rootCommand {
	c := new(rootCommand)
	c.Use = "deploy"
	c.Short = "自动化部署命令"
	c.Long = `自动化部署环境设置工具，包含以下功能：
1. 自动配置Python环境(pip源设置为清华镜像)
2. 安装基础软件包(git, cmake等)
3. 安装机器学习相关依赖(PyTorch, TensorFlow等)
4. 克隆LLaMA-Factory仓库

参数说明:
  -h, --help   显示帮助信息

环境要求:
  - Linux系统(推荐Ubuntu)
  - Python 3.8+
  - pip 最新版本`
	c.Example = `
示例用法:
1. 查看帮助:
   deploy -h

2. 运行完整部署流程:
   deploy

3. 仅执行特定步骤(开发中):
   deploy --step=1-5

4. 程序第一次执行完毕
	需要进入到HHXY_PFAI\cs6493nlp\qgevalcap目录下
	再次执行安装环境命令
	安装完成后
	在该目录下再次执行命令
	即可进入系统环境安装界面
`
	return c
}

func (c *rootCommand) Run(args []string) error {
	if !confirmDeployment() {
		printColorful("脚本已退出。", "91")
		return nil
	}

	totalSteps := 14
	currentStep := 1

	printColorful("当前操作系统: "+runtime.GOOS, "33")
	progressBar(currentStep, totalSteps, "开始部署流程")
	currentStep++
	// 设置pip源为清华大学镜像
	runCommand([]string{"pip", "config", "set", "global.index-url", "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"}, "设置pip源为清华大学镜像", "34")
	progressBar(currentStep, totalSteps, "设置pip源完成")
	currentStep++
	// 更新apt包索引
	runCommand([]string{"sudo", "apt", "update", "-y"}, "更新apt包索引", "32")
	// 安装基础软件包
	runCommand([]string{"sudo", "apt", "install", "-y", "git", "git-lfs", "cmake", "make", "iftop", "atop"}, "安装基础软件包", "32")
	// 安装huggingface_hub
	runCommand([]string{"pip", "install", "-U", "huggingface_hub"}, "安装huggingface_hub", "34")
	// 设置环境变量
	os.Setenv("HF_ENDPOINT", "https://hf-mirror.com")
	printColorful("下载Qwen/Qwen2-7B模型 如果等待时间较长可自行手动下载\n命令1:export HF_ENDPOINT=https://hf-mirror.com \n命令2:cd /mnt/workspace/HHXY_PFAI/cs6493nlp/qgevalcap \n命令3:huggingface-cli download --resume-download <你的模型名字> --local-dir <要存储的路径>", "91")
	// 安装openjdk-17-jdk
	runCommand([]string{"apt", "install", "-y", "openjdk-17-jdk"}, "安装openjdk-17-jdk", "36")
	// 克隆LLaMA-Factory仓库
	if !runCommand([]string{"git", "clone", "https://github.com/hiyouga/LLaMA-Factory.git", "LLaMA-Factory"}, "克隆LLaMA-Factory仓库", "36") {
		printColorful("克隆仓库失败，程序终止", "91")
		return nil
	}
	// 复制PFAI目录
	srcDir := filepath.Join(filepath.Dir(os.Args[0]), "..", "PFAI")
	dstDir := "./LLaMA-Factory/PFAI/"
	if err := copyDir(srcDir, dstDir); err != nil {
		printColorful(fmt.Sprintf("复制目录失败: %v", err), "91")
		return nil
	}

	// 复制dataset_info.json文件
	srcFile := filepath.Join(filepath.Dir(os.Args[0]), "dataset_info.json")
	dstPath := filepath.Join(filepath.Dir(os.Args[0]), "LLaMA-Factory", "data")
	if err := os.MkdirAll(dstPath, 0755); err != nil {
		printColorful(fmt.Sprintf("创建目录失败: %v", err), "91")
		return nil
	}
	if err := copyFile(srcFile, filepath.Join(dstPath, filepath.Base(srcFile))); err != nil {
		printColorful(fmt.Sprintf("复制文件失败: %v", err), "91")
		return nil
	}
	// 安装requirements.txt依赖
	runCommand([]string{"pip", "install", "-r", "LLaMA-Factory/requirements.txt"}, "安装requirements.txt中列出的依赖", "35")
	// 安装额外Python依赖
	runCommand([]string{"pip", "install", "transformers_stream_generator", "bitsandbytes", "tiktoken", "auto-gptq", "optimum", "autoawq"}, "安装额外的Python依赖", "34")
	runCommand([]string{"pip", "install", "--upgrade", "tensorflow"}, "升级tensorflow", "32")
	runCommand([]string{"pip", "install", "vllm==0.4.3"}, "安装vllm", "33")
	runCommand([]string{"pip", "install", "torch==2.1.2", "torchvision==0.16.2", "torchaudio==2.1.2", "--index-url", "https://download.pytorch.org/whl/cu121"}, "安装特定版本的PyTorch", "36")
	runCommand([]string{"pip", "install", "tensorflow==2.12.0"}, "安装tensorflow 2.12.0", "35")
	runCommand([]string{"pip", "install", "-e", "LLaMA-Factory/[metrics]"}, "安装LLaMA-Factory的metrics模块", "32")
	runCommand([]string{"pip", "install", "ijson", "pyaml"}, "安装额外的Python依赖", "34")
	// 克隆nltk_data
	if err := cloneNltkData(); err != nil {
		printColorful(fmt.Sprintf("克隆nltk_data失败: %v", err), "91")
	}

	progressBar(totalSteps, totalSteps, "所有步骤完成")
	printColorful("自动化部署流程已完成。", "32")
	return nil
}

func checkProjectStructure() bool {
	requiredDirs := []string{
		"cs6493nlp/qgevalcap",
		"LLaMA-Factory",
		"go",
	}

	// 检查HHXY_PFAI目录是否存在
	if _, err := os.Stat("HHXY_PFAI"); err == nil {
		// 如果HHXY_PFAI存在，严格检查所有必需子目录
		allDirsExist := true
		for _, dir := range requiredDirs {
			if _, err := os.Stat(filepath.Join("HHXY_PFAI", dir)); os.IsNotExist(err) {
				allDirsExist = false
				break
			}
		}
		if !allDirsExist {
			return false
		}
		return true
	}

	// 检查HHXY_PFAI_STU目录是否存在
	if _, err := os.Stat("HHXY_PFAI_STU"); err == nil {
		return true
	}

	// 检查当前目录结构
	for _, dir := range requiredDirs {
		if _, err := os.Stat(dir); os.IsNotExist(err) {
			return false
		}
	}
	return true
}

func downloadAndSetupProject() error {
	printColorful("警告：请使用root权限运行此程序以确保所有操作能正常执行。", "91")
	printColorful("注意：此程序将下载约120MB的项目文件并安装必要的依赖环境。", "33")

	// 检查HHXY_PFAI_STU目录是否已存在
	if _, err := os.Stat("HHXY_PFAI_STU"); err == nil {
		printColorful("HHXY_PFAI_STU目录已存在，跳过克隆步骤", "33")
		return nil
	}

	printColorful("正在从GitHub下载项目源码...", "34")
	if !runCommand([]string{"git", "clone", "--filter=blob:none", "https://github.com/xiaorui3/HHXY_PFAI_STU.git", "HHXY_PFAI_STU"}, "克隆项目仓库(跳过LFS文件)", "34") {
		//return fmt.Errorf("克隆项目失败")
	}

	// 确保目标目录存在
	targetDir := filepath.Join("HHXY_PFAI_STU", "cs6493nlp", "qgevalcap")
	if err := os.MkdirAll(targetDir, 0755); err != nil {
		return fmt.Errorf("创建目标目录失败: %v", err)
	}

	// 获取当前可执行文件路径
	exePath, err := os.Executable()
	if err != nil {
		return fmt.Errorf("获取可执行文件路径失败: %v", err)
	}

	// 复制当前程序到目标目录
	targetPath := filepath.Join(targetDir, filepath.Base(exePath))
	if err := copyFile(exePath, targetPath); err != nil {
		return fmt.Errorf("复制可执行文件失败: %v", err)
	}

	// 移动临时目录到项目目录
	if _, err := os.Stat("HHXY_PFAI_STU"); os.IsNotExist(err) {
		return fmt.Errorf("源目录HHXY_PFAI_STU不存在")
	}
	if _, err := os.Stat("HHXY_PFAI"); err == nil {
		printColorful("警告：目标目录HHXY_PFAI已存在，将继续执行文件复制操作", "33")
	}
	// 检查HHXY_PFAI是否存在且是文件还是目录
	if fi, err := os.Stat("HHXY_PFAI"); err == nil {
		if !fi.IsDir() {
			// 如果是文件则删除
			if err := os.Remove("HHXY_PFAI"); err != nil {
				return fmt.Errorf("删除文件失败: %v", err)
			}
			if err := os.Rename("HHXY_PFAI_STU", "HHXY_PFAI"); err != nil {
				return fmt.Errorf("重命名目录失败: %v", err)
			}
		} else {
			// 如果是目录则直接使用
			printColorful("警告：HHXY_PFAI目录已存在，将继续使用现有目录", "33")
		}
	} else {
		// 不存在则重命名
		if err := os.Rename("HHXY_PFAI_STU", "HHXY_PFAI"); err != nil {
			return fmt.Errorf("重命名目录失败: %v", err)
		}
	}

	absPath, _ := filepath.Abs(targetPath)
	// 替换路径中的HHXY_PFAI_STU为HHXY_PFAI
	correctedPath := strings.Replace(absPath, "HHXY_PFAI_STU", "HHXY_PFAI", 1)
	printColorful(fmt.Sprintf("项目已设置完成，请运行以下命令:\n%s", correctedPath), "32")
	printColorful("注意：由于HHXY_PFAI目录已存在，部分文件可能已更新，请检查是否需要覆盖。", "33")
	return nil
}

func main() {
	if !checkProjectStructure() {
		printColorful("当前目录结构不符合HHXY_PFAI项目要求", "91")
		if err := downloadAndSetupProject(); err != nil {
			printColorful(fmt.Sprintf("自动设置失败: %v", err), "91")
			return
		}
		return
	}

	cli.NewApplication(newRootCommand).
		SetProperty(app.BannerDisabled, true).
		Run()
}
