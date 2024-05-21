### KNN (k-nearest neighbors)

This is a simple implementation of KNN classifier for continuous data in **C++**,


#### Compilation instruction:  ( For Linux / OSX )
**1.** Compile and Run with these below commands:
( Assuming current directroy is the directory where the classify.cpp file is, if not then goto section "2" )

```bash
~ g++ classify.cpp -o classify
```

#### For Training and Test Prediction

```bash
~ ./classify [training_file] [test_file] knn [value_of_k]
```


#### Example

```bash
~ ./classify data/yeast_training.txt data/yeast_test.txt knn 19
```


**2.** To check current Directory Use this command :
```bash
~ pwd
```
Change the directory to Decision Tree :

```bash
~ cd KNN_DIRECTORIES_PATH
```

Example:
```bash
~ cd /home/USER/MachineLearning/KNN
```
Now Go back to section "**1**"



**Compiler version**
Some header files requires **C++11** or higher version of gcc compiler. ( **e.g.** unordered_map, utility, vector, queue, tuple )

These are only available in C++11 compilation mode, i.e. -std=c++11 or -std=gnu++11.






#### Reference :

[https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_headers.html](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_headers.html)
