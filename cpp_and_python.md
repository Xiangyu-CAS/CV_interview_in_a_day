# 高频问题
- python进程/线程/协程的区别
- C++多态(继承/重写/重载/虚函数)是什么
- C++11的新特性

# C++
## 继承和多态
- 继承定义: 子类函数可以继承父类函数的功能
- 多态定义：一个类的同名函数，在不同情况下游多种实现，子类重写父类的方法。
- 虚函数virtual: 用于实现多态功能，父类成员函数设定为虚函数后，子类可以重写该函数，实现不同的功能。
- 构造函数不能是虚函数，析构函数可以是虚函数。因为构造函数时用来生成对象的，而虚函数是基于对象的扩展，另一方面，从内存的角度来讲
  
## C++11的新特性
- 智能指针。shared_ptr，unique_ptr，auto_ptr，weak_ptr
- lambda。
- auto。
- nullptr代替NULL。
- 多线程thread作为标准库，可以直接调用

# python
## 多进程/多线程/协程
- 多线程。在CPU单个核心上互斥锁串行执行，资源共享。特殊的，由于python解释器存在GIL，一个进程当中如果存在多个线程，也只能在单核CPU上顺序执行，不能利用多个CPU，更适合IO密集型。
```python
from threading import Thread

t1 = Thread(target=func, args=())
t2 = Thread(target=func, targs=())
t1.start()
t2.start()
t1.join() # 阻塞线程，等待t1，t2子线程完成
t2.join()

```
```cpp
#include <thread>
#include <mutex>
#include <queue>

using namespace std;
mutex m;
queue<TEMPLATE> Q;
// 生产者
void producer(int a){
    while (true){
        TEMPLATE data;
        m.lock();
        Q.push(data);
        m.unlock();
    }
}
// 消费者
void consumer(int b){
    while (true){
        while (Q.size()){
            m.lock();
            auto data = Q.front();
            Q.pop();
            m.unlock();
        }
    }
}

int main(){
    int a = 0, b = 0;
    thread prod = new thread(producer, a);
    thread coms = new thread(consumer, b);
    prod.join();
    coms.join();
    return 0;
}
```
- 多进程。在多核CPU上并行执行，资源和变量不共享，需要通过管道队列通信，，更适合计算密集型。

```python
from multiprocessing import Process, Queue

p1 = Process(target=func, args=())
p2 = Process(target=func, args=())
p1.start()
p2.start()
p1.join()
p2.join()
```
- 协程。python3.5以后引入的标准库，用于代替多线程的缺点

## 其他问题
- python没有main函数也能执行，为什么需要main函数
`