### Decision Tree

This is a simple implementation of decision tree classifier for continuous data based on **Infomation Gain** in **C++**,

For further explanation on Decision Tree have a look on [this article](https://redwandipto.github.io/blog/2017/12/07/Decision-Tree-Learning/).

#### Compilation instruction:  ( For Linux / OSX )
**1.** Compile and Run with these below commands:
( Assuming current directroy is the directory where the dTree.cpp file is, if not then goto section "2" )

```bash
~ g++ dTree.cpp -o dTree
```

#### For Training and Test Prediction

```bash
~ ./dTree [training_file_path] [test_file_path] [option] [pruning_threshold]
```



#### Example

```bash
~ ./dTree data/yeast_training.txt data/yeast_test.txt optimized 31
```


**2.** To check current Directory Use this command :
```bash
~ pwd
```
Change the directory to Decision Tree :

```bash
~ cd DECISIONTREE_DIRECTORIES_PATH
```

Example:
```bash
~ cd /home/USER/MachineLearning/DecisionTree
```
Now Go back to section "**1**"



**Compiler version**
Some header files requires **C++11** or higher version of gcc compiler. ( **e.g.** unordered_map, utility, vector, queue, tuple )

These are only available in C++11 compilation mode, i.e. -std=c++11 or -std=gnu++11.






#### Reference :

[https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_headers.html](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_headers.html)
