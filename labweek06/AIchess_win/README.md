# 期中作业--中国象棋



#### 文件介绍

* `main.py`是main函数的程序，直接运行这个文件可以实现人机博弈对抗。

* 其他`.py`文件都是程序运行所需要的类，包括`ChessBoard`、`Game`等。

* `images`文件夹是可视化界面所需的图片。
* 对手AI在`ChessAI.py`中实现，对手AI类已被`pyarmor`加密，需要安装`pyarmor`库才能运行此py文件。另外，我们提供了`linux`、`windows`、`mac`三个版本的加密文件，根据自己电脑的系统选择对应版本的程序代码。
* `MyAI.py`提供了`ChessAI.py`中部分代码逻辑，其中包括了`Evaluate`、`ChessMap`、`ChessAI`三个类。`Evaluate`类提供了当前象棋局面的奖励值，即每个棋子在棋盘上的任意位置都会有一个奖励值，所有棋子的奖励值之和为整个棋面的奖励值。`ChessAI`是实现算法的核心类，须在此类中实现搜索算法。
* 最终评估方法：与对手AI共博弈2次，其中先手、后手各评估一次（`在main.py中未实现算法的红黑机指定代码，需自行实现`）。



#### 代码运行

建议使用`python3.7或python3.6`运行代码

需要安装`pygame`、`numpy`、`pyarmor`库：

```
pip install pygame numpy pyarmor
```

开始程序的命令：

``` python
# 在terminal中运行：
python main.py
# 在pycharm或vscode中运行：
main.py
```



## 提示

* 重复走棋子：重复走子，判输（已实现 judge_draw）
* 和棋：如果30个回合没有棋子被吃，判和（已实现 judge_draw）
* 目前给出的代码版本，我方落子是通过鼠标点击实现的，可以仿照ai落子的代码，修改main，实现MyAI的落子。
* 博弈树搜索：实现在MyAI的ChessAI.get_next_step()中，通过搜索得到下一步落子。

