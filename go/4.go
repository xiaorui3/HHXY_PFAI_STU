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
	if !runCommand([]string{"git", "clone", "https://github.com/nltk/nltk_data.git"}, "克隆nltk_data仓库", "34") {
		return fmt.Errorf("克隆nltk_data仓库失败")
	}

	// 设置目标目录
	targetDir := "/root/nltk_data"
	if err := os.MkdirAll(targetDir, 0755); err != nil {
		return fmt.Errorf("创建目标目录失败: %v", err)
	}

	// 复制packages目录内容
	srcDir := filepath.Join("nltk_data", "packages")
	if _, err := os.Stat(srcDir); os.IsNotExist(err) {
		printColorful("nltk_data项目中不存在packages目录", "91")
	} else {
		entries, err := os.ReadDir(srcDir)
		if err != nil {
			return fmt.Errorf("读取packages目录失败: %v", err)
		}

		for _, entry := range entries {
			srcPath := filepath.Join(srcDir, entry.Name())
			dstPath := filepath.Join(targetDir, entry.Name())

			if entry.IsDir() {
				if err := copyDir(srcPath, dstPath); err != nil {
					return fmt.Errorf("复制目录失败: %v", err)
				}
			} else {
				if err := copyFile(srcPath, dstPath); err != nil {
					return fmt.Errorf("复制文件失败: %v", err)
				}
			}
		}
	}

	// 清理临时目录
	if err := os.RemoveAll("nltk_data"); err != nil {
		printColorful(fmt.Sprintf("清理临时目录失败: %v", err), "91")
	}

	// 解压zip文件
	filesToExtract := []string{"punkt.zip", "punkt_tab.zip"}
	extractDir := filepath.Join(targetDir, "tokenizers")
	if err := os.MkdirAll(extractDir, 0755); err != nil {
		return fmt.Errorf("创建解压目录失败: %v", err)
	}

	for _, fileName := range filesToExtract {
		filePath := filepath.Join(extractDir, fileName)
		if _, err := os.Stat(filePath); err == nil {
			printColorful(fmt.Sprintf("正在解压: %s", fileName), "34")
			if err := unzip(filePath, extractDir); err != nil {
				printColorful(fmt.Sprintf("解压文件失败: %v", err), "91")
			}
		} else {
			printColorful(fmt.Sprintf("文件不存在，跳过解压: %s", filePath), "33")
		}
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
	if !runCommand([]string{"git", "clone", "https://github.com/hiyouga/LLaMA-Factory.git"}, "克隆LLaMA-Factory仓库", "36") {
		printColorful("克隆仓库失败，程序终止", "91")
		return nil
	}
	// 复制PFAI目录
	srcDir := "../../LLaMA-Factory/PFAI/"
	dstDir := "./LLaMA-Factory/PFAI/"
	if err := copyDir(srcDir, dstDir); err != nil {
		printColorful(fmt.Sprintf("复制目录失败: %v", err), "91")
		return nil
	}

	// 复制dataset_info.json文件
	srcFile := "./dataset_info.json"
	dstPath := "./LLaMA-Factory/data/"
	if err := os.MkdirAll(dstPath, 0755); err != nil {
		printColorful(fmt.Sprintf("创建目录失败: %v", err), "91")
		return nil
	}
	if err := copyFile(srcFile, filepath.Join(dstPath, filepath.Base(srcFile))); err != nil {
		printColorful(fmt.Sprintf("复制文件失败: %v", err), "91")
		return nil
	}
	// 安装requirements.txt依赖
	runCommand([]string{"pip", "install", "-r", "requirements.txt"}, "安装requirements.txt中列出的依赖", "35")
	// 安装额外Python依赖
	runCommand([]string{"pip", "install", "transformers_stream_generator", "bitsandbytes", "tiktoken", "auto-gptq", "optimum", "autoawq"}, "安装额外的Python依赖", "34")
	runCommand([]string{"pip", "install", "--upgrade", "tensorflow"}, "升级tensorflow", "32")
	runCommand([]string{"pip", "install", "vllm==0.4.3"}, "安装vllm", "33")
	runCommand([]string{"pip", "install", "torch==2.1.2", "torchvision==0.16.2", "torchaudio==2.1.2", "--index-url", "https://download.pytorch.org/whl/cu121"}, "安装特定版本的PyTorch", "36")
	runCommand([]string{"pip", "install", "tensorflow==2.12.0"}, "安装tensorflow 2.12.0", "35")
	runCommand([]string{"pip", "install", "-e", ".[metrics]"}, "安装LLaMA-Factory的metrics模块", "32")
	runCommand([]string{"pip", "install", "ijson", "pyaml"}, "安装额外的Python依赖", "34")
	// 克隆nltk_data
	if err := cloneNltkData(); err != nil {
		printColorful(fmt.Sprintf("克隆nltk_data失败: %v", err), "91")
	}

	progressBar(totalSteps, totalSteps, "所有步骤完成")
	printColorful("自动化部署流程已完成。", "32")
	return nil
}

func main() {
	cli.NewApplication(newRootCommand).
		SetProperty(app.BannerDisabled, true).
		Run()
}
